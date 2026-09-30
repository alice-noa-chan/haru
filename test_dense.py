"""Numerical, resume, export and budget guarantees for the new architecture."""

from __future__ import annotations

import copy
import importlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from configuration_haru_dense import HaruDenseConfig
from data_utils import blake2b_file
from dense_model import CANDIDATES, DenseConfig, DenseLanguageModel, grow_teacher, parameter_count, student_from_teacher
from haru.archive import bundle, unpack
from haru.data import (
    CHAT_TEMPLATE,
    NAMES,
    ROLES,
    STORY_CHAT_WEIGHTS,
    STORY_WEIGHTS,
    WEIGHTS,
    MixtureSampler,
    canonical,
    data_profile,
    rule_examples,
    split_for,
)
from haru.runtime import Deadline, atomic_json
from haru.training import (
    ChatSampler,
    checkpoint_validation,
    distillation_loss,
    evaluate_training,
    load_model,
    parser,
    save_checkpoint,
    train,
)
from modeling_haru_dense import HaruDenseForCausalLM
from tokenization_cfrd import CFRDTokenizer


class DenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tiny(self, gated=False):
        cfg = DenseConfig(
            vocab_size=71,
            d_model=32,
            n_head=2,
            n_kv_head=1,
            ffn_dim=64,
            n_layer=2,
            attention_gate=gated,
            context_length=1024,
        )
        return DenseLanguageModel(cfg, torch.randn(71, 76))

    def test_pipeline_stops_after_completed_teacher(self):
        from haru.pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "overlay"
            overlay.mkdir()
            (overlay / "rule_generation.json").write_text("{}")
            args = SimpleNamespace(
                output=root / "runs",
                data=root / "data",
                teacher_rules_overlay=overlay,
                device="cpu",
                deadline_unix=None,
                max_seconds=None,
                teacher_tokens=2_000_000_000,
                student_tokens=500_000_000,
                chat_tokens=10_000_000,
                stop_after="teacher-base",
            )
            metrics = {
                "macro_bpc": 1.0,
                "selection_score": 1.0,
                "domains": {"story": {"bpc": 1.0, "document_bpc": [1.0]}},
            }

            def choose(rows):
                ranking = sorted(rows, key=lambda row: row["candidate"] != "gated8")
                return ranking[0], ranking

            with (
                patch("haru.pipeline.train", return_value={"status": "complete"}) as training,
                patch("haru.pipeline.checkpoint_validation", side_effect=lambda _: copy.deepcopy(metrics)),
                patch("haru.pipeline.select", side_effect=choose),
                patch("haru.pipeline.materialize", return_value=root / "teacher-data"),
                patch("haru.pipeline.export") as exporting,
            ):
                result = run_pipeline(args)
            self.assertEqual(result["status"], "phase_complete")
            self.assertEqual(result["phase"], "teacher-base")
            phases = [call.args[0].phase for call in training.call_args_list]
            self.assertEqual(phases[-1], "teacher-base")
            self.assertNotIn("student-base", phases)
            exporting.assert_called_once()

    def test_exact_parameter_cap(self):
        expected = {"dense8": 17_227_649, "deep10": 17_389_121, "gated8": 17_817_473}
        for name, cfg in CANDIDATES.items():
            with torch.device("meta"):
                model = DenseLanguageModel(cfg, torch.zeros(12000, 76))
            actual = sum(p.numel() for p in model.parameters())
            self.assertEqual(actual, expected[name])
            self.assertEqual(actual, parameter_count(cfg))
            self.assertLess(actual, 18_000_000)

    def test_training_bundle_hash_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "manifest.json").write_text('{"fixture":true}')
            (source / "web.train.bin").write_bytes(bytes(range(256)) * 4)
            (source / "chat.train.jsonl").write_text("{}\n")
            (source / "web.train.jsonl").write_text("unnecessary training documents")
            (source / "web.val.jsonl").write_text("validation documents")
            packed = root / "data.tar.zst"
            manifest = bundle(source, packed, training_data=True)
            self.assertNotIn("web.train.jsonl", manifest["files"])
            self.assertIn("web.val.jsonl", manifest["files"])
            destination = root / "unpacked"
            unpack(packed, destination)
            self.assertEqual((destination / "web.train.bin").read_bytes(), (source / "web.train.bin").read_bytes())
            record = packed.with_name(packed.name + ".json")
            manifest["files"]["web.train.bin"]["blake2b16"] = "wrong"
            record.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                unpack(packed, root / "corrupt")
            packed.write_bytes(packed.read_bytes()[:-10])
            with self.assertRaisesRegex(ValueError, "Bundle hash mismatch"):
                unpack(packed, root / "truncated")

    def test_causality_and_gradients(self):
        for gated in (False, True):
            model = self.tiny(gated)
            ids = torch.randint(0, 71, (2, 65))
            changed = ids.clone()
            changed[:, 32:] = torch.randint(0, 71, changed[:, 32:].shape)
            torch.testing.assert_close(model(ids).logits[:, :32], model(changed).logits[:, :32])
            model(ids, targets=ids).loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_cache_boundaries_and_padding(self):
        for gated in (False, True):
            model = self.tiny(gated).eval()
            for length in (1, 63, 64, 65, 512, 1024):
                ids = torch.randint(0, 71, (2, length))
                with torch.no_grad():
                    full = model(ids).logits
                    prefix = max(1, length // 2)
                    cached = model(ids[:, :prefix], use_cache=True)
                    if prefix < length:
                        tail = model(ids[:, prefix:], past_key_values=cached.past_key_values, use_cache=True)
                        torch.testing.assert_close(full[:, prefix:], tail.logits, atol=1e-5, rtol=1e-5)
                        cached = tail
                    with self.assertRaises(ValueError):
                        model(
                            ids[:, :1],
                            past_key_values=model(
                                torch.zeros(2, 1024, dtype=torch.long), use_cache=True
                            ).past_key_values,
                        )
            ids = torch.randint(1, 71, (2, 65))
            mask = torch.ones_like(ids)
            mask[0, :7] = 0
            ids[0, :7] = 0
            with torch.no_grad():
                full = model(ids, attention_mask=mask)
                compact = model(ids[0:1, 7:]).logits
                torch.testing.assert_close(full.logits[0:1, 7:], compact, atol=1e-5, rtol=1e-5)
                cached = model(ids[:, :32], attention_mask=mask[:, :32], use_cache=True)
                tail = model(ids[:, 32:], attention_mask=mask, past_key_values=cached.past_key_values, use_cache=True)
                torch.testing.assert_close(full.logits[:, 32:], tail.logits, atol=1e-5, rtol=1e-5)

    def test_teacher_identity_and_student_copy(self):
        for gated in (False, True):
            model = self.tiny(gated).eval()
            teacher = grow_teacher(model).eval()
            ids = torch.randint(0, 71, (2, 63))
            torch.testing.assert_close(model(ids).logits, teacher(ids).logits, atol=0, rtol=0)
            restored = student_from_teacher(teacher).eval()
            torch.testing.assert_close(model(ids).logits, restored(ids).logits, atol=0, rtol=0)

    def test_hf_wrapper_finalizes_without_changing_dense_initialization(self):
        cfg = DenseConfig(
            vocab_size=71,
            d_model=32,
            n_head=2,
            n_kv_head=1,
            ffn_dim=64,
            n_layer=2,
            ffn_share_group_size=2,
            ffn_adapter_rank=8,
        )
        torch.manual_seed(197)
        expected = DenseLanguageModel(cfg, torch.zeros(cfg.vocab_size, cfg.surface_feature_dim))
        torch.manual_seed(197)
        wrapper = HaruDenseForCausalLM(HaruDenseConfig(dense_config=asdict(cfg)))
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, wrapper.model.state_dict()[name]), name)
        if hasattr(wrapper, "all_tied_weights_keys"):
            self.assertEqual(wrapper.all_tied_weights_keys, {})
        # Use the real version's save/load path so missing finalization metadata regresses.
        with tempfile.TemporaryDirectory() as directory:
            wrapper.save_pretrained(directory)
            loaded = HaruDenseForCausalLM.from_pretrained(directory)
            for name, value in wrapper.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)

    def test_remote_inference_import_check_without_optional_liger(self):
        from transformers.dynamic_module_utils import check_imports

        original = importlib.import_module

        def without_liger(name, *args, **kwargs):
            if name.startswith("liger_kernel"):
                raise ImportError("Optional training kernel is unavailable")
            return original(name, *args, **kwargs)

        with patch("importlib.import_module", side_effect=without_liger):
            check_imports(str(Path(__file__).with_name("dense_model.py")))

    def test_kd_and_assistant_mask(self):
        model = self.tiny()
        loss_only = copy.deepcopy(model)
        ids = torch.randint(0, 71, (2, 8))
        gold = ids.clone()
        gold[:, :5] = -100
        full = model(ids, targets=gold)
        compact = loss_only(ids, targets=gold, loss_only=True)
        self.assertIsNone(compact.logits)
        torch.testing.assert_close(full.loss, compact.loss)
        full.loss.backward()
        compact.loss.backward()
        for original, optimized in zip(model.parameters(), loss_only.parameters()):
            torch.testing.assert_close(original.grad, optimized.grad)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            model(ids, targets=gold, loss_only=True, loss_backend="liger")
        student = torch.randn(2, 8, 71, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        targets = torch.randint(0, 71, (2, 8))
        targets[:, :5] = -100
        expected = distillation_loss(student, teacher, targets)
        changed = student.detach().clone()
        changed[:, :5] += 100
        torch.testing.assert_close(expected, distillation_loss(changed, teacher, targets))
        expected.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(student.grad[:, :5].abs().sum().item(), 0)

    def test_atomic_checkpoint_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            model = self.tiny()
            optimizer = torch.optim.AdamW(model.parameters())
            sampler = type("Sampler", (), {"state_dict": lambda self: {"cursor": 7}})()
            save_checkpoint(path, model, optimizer, sampler, 3, 100, 2.0, {}, {})
            before = blake2b_file(path)
            with patch("torch.save", side_effect=OSError("disk full")), self.assertRaises(OSError):
                save_checkpoint(path, model, optimizer, sampler, 4, 120, 1.0, {}, {})
            self.assertEqual(before, blake2b_file(path))
            loaded, state = load_model(path)
            self.assertEqual(state["step"], 3)
            self.assertEqual(state["sampler"]["cursor"], 7)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]))
            metrics = {"step": 3, "tokens_seen": 100, "macro_bpc": 2.0}
            best = path.parent / "best.pt"
            save_checkpoint(best, model, optimizer, sampler, 3, 100, 2.0, {}, {"selection_validation": metrics})
            atomic_json(best.parent / "validation_best.json", {**metrics, "step": 4, "macro_bpc": 1.0})
            self.assertEqual(checkpoint_validation(best), metrics)
            del state["model"]
            state["selection_validation"] = {**metrics, "step": 4}
            with self.assertRaisesRegex(ValueError, "do not belong"):
                checkpoint_validation(best, state)

    def test_deadline_boundary(self):
        now = [0.0]
        deadline = Deadline(seconds=5, clock=lambda: now[0])
        self.assertFalse(deadline.expired())
        now[0] = 5
        self.assertTrue(deadline.expired())

    def test_atomic_json_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            original = Path.replace
            calls = [0]

            def transient_lock(source, destination):
                calls[0] += 1
                if calls[0] < 3:
                    raise PermissionError("temporary file watcher lock")
                return original(source, destination)

            with patch.object(Path, "replace", transient_lock), patch("haru.runtime.time.sleep"):
                atomic_json(path, {"status": "saved"})
            self.assertEqual(json.loads(path.read_text()), {"status": "saved"})
            self.assertEqual(calls[0], 3)
        deadline = Deadline(seconds=5, clock=lambda: 0)
        deadline.request_stop()
        self.assertTrue(deadline.expired())

    def test_document_and_factor_splits(self):
        self.assertEqual(canonical("안녕, 세상!"), canonical("안녕 세상"))
        self.assertEqual(split_for("12345678"), split_for("12345678"))
        sets = [set(values) for values in NAMES.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        rendering = []
        for split in NAMES:
            items = list(rule_examples(700, split))
            rendering.append({x["render_family"] for x in items})
            self.assertEqual(
                {x["task"] for x in items},
                {"location", "state", "ownership", "transfer", "speaker", "negation", "arithmetic"},
            )
            for item in items:
                self.assertEqual(item["text"], item["prompt"] + item["answer"])
            self.assertEqual({x["answer"] for x in items if x["task"] == "state"}, {"열린 상태", "닫힌 상태"})
            self.assertGreater(len({x["answer"] for x in items if x["task"] == "speaker"}), 3)
            self.assertGreater(len({tuple(x["event_order"]) for x in items if x["task"] == "location"}), 3)
        self.assertFalse(rendering[0] & rendering[1] or rendering[0] & rendering[2] or rendering[1] & rendering[2])

    def test_exact_sampler_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "manifest.json").write_text(json.dumps({"weights": WEIGHTS}))
            for domain in WEIGHTS:
                np.arange(4096, dtype=np.uint16).tofile(Path(directory) / f"{domain}.train.bin")
            original = MixtureSampler(directory, 1337)
            grouped = MixtureSampler(directory, 1337)
            entire = original.batch(4, 63)
            pieces = [grouped.batch(2, 63), grouped.batch(2, 63)]
            for actual, expected_rows in zip(entire, (np.concatenate([p[i] for p in pieces]) for i in (0, 1))):
                np.testing.assert_array_equal(actual, expected_rows)
            grouped.close()
            original.batch(4, 63)
            state = copy.deepcopy(original.state_dict())
            expected = original.batch(4, 63)
            resumed = MixtureSampler(directory, 918)
            resumed.load_state_dict(state)
            actual = resumed.batch(4, 63)
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(expected, actual, strict=True)))
            original.close()
            resumed.close()

    def test_story_view_and_story_first_selection(self):
        from haru.expand_rules import materialize
        from haru.pipeline import select
        from haru.story_view import materialize_story_view

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            base.mkdir()
            (base / "tokenizer.model").write_bytes(b"fixed tokenizer")
            streams = {}
            for domain in WEIGHTS:
                path = base / f"{domain}.train.bin"
                np.arange(4096, dtype=np.uint16).tofile(path)
                streams[f"{domain}.train"] = {"tokens": 4096, "blake2b16": blake2b_file(path)}
            original = {
                "weights": WEIGHTS,
                "tokenizer_blake2b16": blake2b_file(base / "tokenizer.model"),
                "streams": streams,
            }
            (base / "manifest.json").write_text(json.dumps(original))
            original_hash = blake2b_file(base / "manifest.json")
            story = materialize_story_view(base, Path(directory) / "story")
            self.assertEqual(materialize_story_view(base, story), story)
            self.assertEqual(original_hash, blake2b_file(base / "manifest.json"))
            profile, weights, chat_weights = data_profile(story)
            self.assertEqual(profile["objective"], "story_continuation")
            self.assertEqual(weights, STORY_WEIGHTS)
            self.assertEqual(chat_weights, STORY_CHAT_WEIGHTS)
            sampler = MixtureSampler(story)
            self.assertEqual(sampler.probabilities[sampler.domains.index("story")], 0.5)
            sampler.close()
            validation = {
                "macro_bpc": 9.0,
                "domains": {"story": {"bpc": 1.5, "document_bpc": [1.5], "document_characters": [100]}},
            }
            with patch("haru.training.evaluate", return_value=validation):
                result = evaluate_training(None, None, SimpleNamespace(data=story, eval_documents=1, chat=False))
            self.assertEqual(result["selection_metric"], "story_bpc")
            self.assertEqual(result["selection_score"], 1.5)
            rows = [
                {"validation": {**result, "macro_bpc": 2.0}, "checkpoint": "a"},
                {
                    "validation": {
                        **result,
                        "selection_score": 2.0,
                        "macro_bpc": 1.0,
                        "domains": {"story": {"bpc": 2.0, "document_bpc": [2.0], "document_characters": [100]}},
                    },
                    "checkpoint": "b",
                },
            ]
            winner, _ = select(rows)
            self.assertEqual(winner["checkpoint"], "a")
            overlay = Path(directory) / "overlay"
            overlay.mkdir()
            (overlay / "rules.train.bin").write_bytes(b"expanded rules")
            (overlay / "rule_generation.json").write_text(
                json.dumps(
                    {
                        "base_manifest_blake2b16": original_hash,
                        "tokenizer_blake2b16": original["tokenizer_blake2b16"],
                        "rules.train": {"tokens": 7, "blake2b16": blake2b_file(overlay / "rules.train.bin")},
                    }
                )
            )
            teacher = materialize(story, overlay, Path(directory) / "teacher")
            self.assertEqual(data_profile(teacher)[0]["objective"], "story_continuation")

    def test_hf_independent_export_chat_and_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample.txt"
            sample.write_text(
                "안녕하세요. 작은 마을에 하린과 수아가 살아요.\n빨간 열쇠가 상자 안에 있어요.\n" * 50, encoding="utf-8"
            )
            spm.SentencePieceTrainer.train(
                input=str(sample),
                model_prefix=str(root / "tokenizer"),
                model_type="bpe",
                vocab_size=128,
                hard_vocab_limit=False,
                normalization_rule_name="identity",
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                user_defined_symbols=["<|nl|>", "<|literal_nl|>", *ROLES],
                minloglevel=2,
            )
            sp = spm.SentencePieceProcessor(model_file=str(root / "tokenizer.model"))
            from haru.expand_rules import expand, materialize

            (root / "manifest.json").write_text('{"streams":{"rules.train":{"tokens":1}}}')
            (root / "rules.train.bin").write_bytes(b"\x01\x00")
            original_data_hash = blake2b_file(root / "manifest.json")
            overlay = root / "overlay"
            record = expand(root, overlay, count=70)
            self.assertEqual(record["unique_examples"], 70)
            self.assertTrue(all(value == 10 for value in record["task_counts"].values()))
            self.assertEqual(expand(root, overlay, count=70), record)
            with self.assertRaisesRegex(ValueError, "different or damaged version"):
                expand(root, overlay, count=77)
            teacher_data = materialize(root, overlay, root / "teacher-data")
            self.assertEqual(original_data_hash, blake2b_file(root / "manifest.json"))
            self.assertEqual((root / "rules.train.bin").read_bytes(), b"\x01\x00")
            self.assertEqual(blake2b_file(root / "tokenizer.model"), blake2b_file(teacher_data / "tokenizer.model"))
            self.assertNotEqual(original_data_hash, blake2b_file(teacher_data / "manifest.json"))
            (overlay / "rules.train.bin").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "stream hash mismatch"):
                materialize(root, overlay, root / "another-view")
            cfg = DenseConfig(vocab_size=sp.vocab_size(), d_model=32, n_head=2, n_kv_head=1, ffn_dim=64, n_layer=2)
            wrapper = HaruDenseForCausalLM(HaruDenseConfig(dense_config=asdict(cfg))).eval()
            tokenizer = CFRDTokenizer(
                str(root / "tokenizer.model"),
                bos_token="<s>",
                eos_token="</s>",
                pad_token="<pad>",
                unk_token="<unk>",
                additional_special_tokens=ROLES,
                extra_special_tokens={},
                chat_template=CHAT_TEMPLATE,
                padding_side="left",
            )
            self.assertEqual(len(tokenizer), cfg.vocab_size)
            chat = [
                {
                    "messages": [
                        {"role": "user", "content": "작은 마을\n하린"},
                        {"role": "assistant", "content": "하린"},
                    ],
                    "task": "grounded_extract",
                },
                {
                    "messages": [{"role": "user", "content": "안녕하세요"}, {"role": "assistant", "content": "수아"}],
                    "task": "story",
                },
            ]
            (root / "chat.train.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in chat), encoding="utf-8"
            )
            (root / "manifest.json").write_text(json.dumps({"weights": WEIGHTS}))
            from tokenizer_utils import StoryTokenizer

            packed = ChatSampler(root, StoryTokenizer(root / "tokenizer.model"), 1337)
            grouped = ChatSampler(root, StoryTokenizer(root / "tokenizer.model"), 1337)
            rows = packed.batch(4, 63)
            pieces = [grouped.batch(2, 63), grouped.batch(2, 63)]
            for i in (0, 1):
                np.testing.assert_array_equal(rows[i], np.concatenate([piece[i] for piece in pieces]))
            self.assertTrue((rows[0] != 0).all())
            self.assertTrue((rows[1] == -100).any())
            saved = packed.state_dict()
            next_rows = packed.batch(2, 63)
            packed.load_state_dict(saved)
            resumed_rows = packed.batch(2, 63)
            np.testing.assert_array_equal(next_rows[0], resumed_rows[0])
            np.testing.assert_array_equal(next_rows[1], resumed_rows[1])
            export = root / "export"
            wrapper.save_pretrained(export)
            tokenizer.save_pretrained(export)
            loaded = AutoModelForCausalLM.from_pretrained(export, trust_remote_code=True, local_files_only=True).eval()
            loaded_tokenizer = AutoTokenizer.from_pretrained(export, trust_remote_code=True, local_files_only=True)
            ids = loaded_tokenizer.apply_chat_template(
                [{"role": "user", "content": "안녕하세요."}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False,
            )
            self.assertEqual(ids[0, -1].item(), loaded_tokenizer.convert_tokens_to_ids("<|assistant|>"))
            self.assertEqual(ids[0, 0].item(), loaded_tokenizer.bos_token_id)
            full_chat = loaded_tokenizer.apply_chat_template(chat[0]["messages"], return_dict=False)
            self.assertEqual(full_chat, packed.items[0][0])
            roles = [loaded_tokenizer.convert_tokens_to_ids(role) for role in ROLES]
            self.assertTrue(set(roles).issubset(set(loaded_tokenizer.all_special_ids)))
            self.assertEqual(loaded_tokenizer.decode(roles, skip_special_tokens=True), "")
            inputs = loaded_tokenizer(["작은 마을", "안녕하세요. 작은 마을"], padding=True, return_tensors="pt")
            from haru.evaluate import diagnose

            for domain in WEIGHTS:
                (root / f"{domain}.test.jsonl").write_text(
                    json.dumps({"text": "안녕하세요. 작은 마을에 하린이 살아요."}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            case = {"category": "fixture", "prompt": "작은 마을", "expected": " 하린", "contradiction": " 수아"}
            with (
                patch("haru.evaluate.GENERATION_PROMPTS", ["작은 마을"]),
                patch("haru.evaluate.build_cases", return_value=[case]),
            ):
                report = diagnose(export, root, documents=1, rules=7, threads=2)
            self.assertTrue(np.isfinite(report["bpc_shared_context"]["macro_bpc"]))
            self.assertEqual(len(report["program_generalization"]), 4)
            self.assertGreater(report["generation"][0]["tokens_per_second"], 0)
            with torch.no_grad():
                torch.testing.assert_close(wrapper(**inputs).logits, loaded(**inputs).logits)
                cached = loaded.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=True)
                plain = loaded.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=False)
                self.assertTrue(torch.equal(cached, plain))
                beams = loaded.generate(**inputs, max_new_tokens=3, num_beams=2, use_cache=True)
                self.assertEqual(beams.shape[0], 2)
                with self.assertRaises(ValueError):
                    loaded.generate(**inputs, max_new_tokens=1024)

    def test_interrupted_training_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample.txt"
            sample.write_text("안녕하세요. 작은 마을에 하린과 수아가 살아요.\n" * 50, encoding="utf-8")
            spm.SentencePieceTrainer.train(
                input=str(sample),
                model_prefix=str(root / "tokenizer"),
                model_type="bpe",
                vocab_size=96,
                hard_vocab_limit=False,
                normalization_rule_name="identity",
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                user_defined_symbols=["<|nl|>", "<|literal_nl|>", *ROLES],
                minloglevel=2,
            )
            sp = spm.SentencePieceProcessor(model_file=str(root / "tokenizer.model"))
            cfg = DenseConfig(
                vocab_size=sp.vocab_size(), d_model=32, n_head=2, n_kv_head=1, ffn_dim=64, n_layer=2, context_length=16
            )
            rng = np.random.default_rng(918)
            for domain in WEIGHTS:
                rng.integers(4, sp.vocab_size(), 4096, dtype=np.uint16).tofile(root / f"{domain}.train.bin")
                (root / f"{domain}.val.jsonl").write_text(
                    json.dumps({"text": "안녕하세요. 작은 마을에 하린이 살아요."}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            (root / "manifest.json").write_text('{"fixture":true}')

            def options(name, tokens):
                return parser().parse_args(
                    [
                        "--data",
                        str(root),
                        "--output",
                        str(root / name),
                        "--device",
                        "cpu",
                        "--microbatch",
                        "2",
                        "--effective-tokens",
                        "64",
                        "--target-tokens",
                        str(tokens),
                        "--schedule-tokens",
                        "256",
                        "--eval-documents",
                        "1",
                    ]
                )

            with patch.dict("haru.training.CANDIDATES", {"dense8": cfg}):
                train(options("whole", 256))
                train(options("resumed", 128))
                train(options("resumed", 256))
            whole = torch.load(root / "whole/latest.pt", weights_only=False)
            resumed = torch.load(root / "resumed/latest.pt", weights_only=False)
            self.assertEqual(whole["step"], resumed["step"])
            self.assertEqual(whole["sampler"], resumed["sampler"])
            for key in whole["model"]:
                self.assertTrue(torch.equal(whole["model"][key], resumed["model"][key]), key)
            with patch.dict("haru.training.CANDIDATES", {"dense8": cfg}):
                stop = Deadline(seconds=0)
                result = train(options("never_started", 256), stop)
                self.assertEqual(result["status"], "budget_stop")
                self.assertFalse((root / "never_started/latest.pt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
