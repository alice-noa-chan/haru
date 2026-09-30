"""Sharing, adapter, distillation and portable loading guarantees for the student experiment."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM

from configuration_haru_dense import HaruDenseConfig
from data_utils import blake2b_file
from dense_model import (
    EXPERIMENTAL_CANDIDATES,
    DenseConfig,
    DenseLanguageModel,
    grow_teacher,
    parameter_count,
    student_from_teacher,
)
from haru.data import ROLES, WEIGHTS
from haru.training import parser, train
from modeling_haru_dense import HaruDenseForCausalLM


class ExperimentalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **kwargs):
        return replace(
            DenseConfig(
                vocab_size=71,
                d_model=32,
                n_head=2,
                n_kv_head=1,
                ffn_dim=64,
                n_layer=4,
                attention_gate=True,
                ffn_share_group_size=2,
                ffn_adapter_rank=8,
            ),
            **kwargs,
        )

    def model(self, **kwargs):
        cfg = self.config(**kwargs)
        return DenseLanguageModel(cfg, torch.randn(cfg.vocab_size, 76))

    def test_exact_matched_parameter_count_and_unique_registration(self):
        for name, cfg in EXPERIMENTAL_CANDIDATES.items():
            with self.subTest(candidate=name), torch.device("meta"):
                model = DenseLanguageModel(cfg, torch.zeros(cfg.vocab_size, 76))
            self.assertEqual(sum(p.numel() for p in model.parameters()), 13_688_705)
            self.assertEqual(parameter_count(cfg), 13_688_705)
        model = self.model()
        self.assertEqual(len(model.shared_ffns), 2)
        self.assertTrue(all(block.ffn is None for block in model.blocks))
        self.assertEqual(len({id(block.attention) for block in model.blocks}), 4)
        parameters = list(model.parameters())
        self.assertEqual(len(parameters), len(list(model.named_parameters(remove_duplicate=False))))
        pointers = [tensor.data_ptr() for tensor in model.state_dict().values()]
        self.assertEqual(len(pointers), len(set(pointers)))
        for cfg in (self.config(ffn_share_group_size=3), self.config(ffn_adapter_rank=33)):
            with self.assertRaises(ValueError):
                cfg.validate()
        with self.assertRaisesRegex(ValueError, "independent FFN"):
            grow_teacher(model)

    def test_shared_gradient_is_sum_of_independent_layer_gradients(self):
        shared = self.model()
        independent = DenseLanguageModel(replace(shared.cfg, ffn_share_group_size=1), shared.surface_feature_table)
        independent.load_state_dict(
            {name: tensor for name, tensor in shared.state_dict().items() if not name.startswith("shared_ffns.")},
            strict=False,
        )
        # Nonzero adapters exercise both layer-specific paths, not just their neutral initialization.
        for i, block in enumerate(shared.blocks):
            torch.nn.init.normal_(block.ffn_adapter_up.weight, std=0.02)
            independent.blocks[i].ffn_adapter_up.load_state_dict(block.ffn_adapter_up.state_dict())
            independent.blocks[i].ffn.load_state_dict(shared.shared_ffns[i // 2].state_dict())
        ids = torch.randint(0, 71, (2, 17))
        a, b = shared(ids, targets=ids), independent(ids, targets=ids)
        torch.testing.assert_close(a.logits, b.logits, atol=0, rtol=0)
        a.loss.backward()
        b.loss.backward()
        for group, ffn in enumerate(shared.shared_ffns):
            for name, parameter in ffn.named_parameters():
                expected = sum(
                    dict(independent.blocks[2 * group + i].ffn.named_parameters())[name].grad for i in (0, 1)
                )
                torch.testing.assert_close(parameter.grad, expected, atol=1e-7, rtol=1e-5)
        for i, block in enumerate(shared.blocks):
            torch.testing.assert_close(
                block.attention.q_proj.weight.grad, independent.blocks[i].attention.q_proj.weight.grad
            )

    def test_zero_adapter_output_and_learning(self):
        model = self.model()
        self.assertTrue(all(torch.count_nonzero(block.ffn_adapter_up.weight) == 0 for block in model.blocks))
        ids = torch.randint(0, 71, (2, 17))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(ids, targets=ids).loss.backward()
        for block in model.blocks:
            self.assertEqual(torch.count_nonzero(block.ffn_adapter_down.weight.grad), 0)
            self.assertGreater(torch.count_nonzero(block.ffn_adapter_up.weight.grad), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(ids, targets=ids).loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(all(torch.count_nonzero(block.ffn_adapter_down.weight.grad) > 0 for block in model.blocks))

    def test_cache_lengths_padding_and_causality(self):
        model = self.model().eval()
        with torch.no_grad():
            for length in (1, 63, 64, 65, 512, 1024):
                ids = torch.randint(1, 71, (2, length))
                mask = torch.ones_like(ids)
                if length > 1:
                    mask[0, : max(1, length // 8)] = 0
                    ids[mask == 0] = 0
                full = model(ids, attention_mask=mask).logits
                prefix = max(1, length // 2)
                cached = model(ids[:, :prefix], attention_mask=mask[:, :prefix], use_cache=True)
                self.assertEqual(len(cached.past_key_values), model.cfg.n_layer)
                self.assertEqual(len({k.data_ptr() for k, _ in cached.past_key_values}), model.cfg.n_layer)
                if prefix < length:
                    tail = model(
                        ids[:, prefix:], attention_mask=mask, past_key_values=cached.past_key_values, use_cache=True
                    )
                    torch.testing.assert_close(full[:, prefix:], tail.logits, atol=1e-5, rtol=1e-5)
                changed = ids.clone()
                changed[:, prefix:] = torch.randint(1, 71, changed[:, prefix:].shape)
                torch.testing.assert_close(full[:, :prefix], model(changed, attention_mask=mask).logits[:, :prefix])

    def test_teacher_initialization_mapping_and_immutability(self):
        cfg = self.config()
        teacher = self.model(n_layer=8, ffn_share_group_size=1, ffn_adapter_rank=0)
        before = copy.deepcopy(teacher.state_dict())
        shared = student_from_teacher(teacher, cfg)
        narrow = student_from_teacher(teacher, replace(cfg, ffn_share_group_size=1, ffn_adapter_rank=0, ffn_dim=32))
        for i, block in enumerate(shared.blocks):
            torch.testing.assert_close(block.attention.q_proj.weight, teacher.blocks[2 * i].attention.q_proj.weight)
            self.assertEqual(torch.count_nonzero(block.ffn_adapter_up.weight), 0)
            torch.testing.assert_close(narrow.blocks[i].ffn.w1.weight, teacher.blocks[2 * i].ffn.w1.weight[:32])
            torch.testing.assert_close(narrow.blocks[i].ffn.w2.weight, teacher.blocks[2 * i].ffn.w2.weight[:, :32])
        for group, ffn in enumerate(shared.shared_ffns):
            torch.testing.assert_close(ffn.w1.weight, teacher.blocks[4 * group].ffn.w1.weight)
        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, teacher.state_dict()[name]))
        with self.assertRaisesRegex(ValueError, "d_model differs"):
            student_from_teacher(teacher, replace(cfg, d_model=64))

    def test_hf_shared_safetensors_independent_load_and_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = HaruDenseForCausalLM(HaruDenseConfig(dense_config=asdict(self.config()))).eval()
            wrapper.save_pretrained(directory, safe_serialization=True)
            self.assertTrue((Path(directory) / "model.safetensors").exists())
            loaded = AutoModelForCausalLM.from_pretrained(
                directory, trust_remote_code=True, local_files_only=True
            ).eval()
            self.assertEqual(loaded.config.dense_config["ffn_share_group_size"], 2)
            self.assertEqual(loaded.config.dense_config["ffn_adapter_rank"], 8)
            for name, tensor in wrapper.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)
            ids = torch.randint(4, 71, (2, 17))
            mask = torch.ones_like(ids)
            mask[0, :3] = 0
            ids[mask == 0] = 0
            with torch.no_grad():
                torch.testing.assert_close(
                    wrapper(ids, attention_mask=mask).logits, loaded(ids, attention_mask=mask).logits
                )
                cached = loaded.generate(ids, attention_mask=mask, max_new_tokens=4, use_cache=True, do_sample=False)
                plain = loaded.generate(ids, attention_mask=mask, max_new_tokens=4, use_cache=False, do_sample=False)
            self.assertTrue(torch.equal(cached, plain))

    def test_distillation_training_resume_is_exact_and_teacher_unchanged(self):
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
            cfg = self.config(vocab_size=sp.vocab_size(), context_length=16)
            teacher = DenseLanguageModel(
                replace(cfg, n_layer=8, ffn_share_group_size=1, ffn_adapter_rank=0), torch.randn(cfg.vocab_size, 76)
            )
            teacher_path = root / "teacher.pt"
            torch.save(
                {
                    "model_arch": "haru-dense",
                    "model_config": asdict(teacher.cfg),
                    "model": teacher.state_dict(),
                    "tokenizer_blake2b16": blake2b_file(root / "tokenizer.model"),
                },
                teacher_path,
            )
            teacher_hash = blake2b_file(teacher_path)
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
                        "--candidate",
                        "pairshare8-r48",
                        "--teacher",
                        str(teacher_path),
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

            with patch.dict("haru.training.EXPERIMENTAL_CANDIDATES", {"pairshare8-r48": cfg}):
                train(options("whole", 256))
                train(options("resumed", 128))
                train(options("resumed", 256))
            whole = torch.load(root / "whole/latest.pt", weights_only=False)
            resumed = torch.load(root / "resumed/latest.pt", weights_only=False)
            self.assertEqual(whole["step"], resumed["step"])
            self.assertEqual(whole["sampler"], resumed["sampler"])
            self.assertEqual(whole["model_config"]["ffn_share_group_size"], 2)
            for name, tensor in whole["model"].items():
                self.assertTrue(torch.equal(tensor, resumed["model"][name]), name)
            for index, state in whole["optimizer"]["state"].items():
                for name, tensor in state.items():
                    self.assertTrue(torch.equal(tensor, resumed["optimizer"]["state"][index][name]))
            self.assertEqual(teacher_hash, blake2b_file(teacher_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
