from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

import config
from baseline_model import BaselineConfig, BaselineLanguageModel
from compare_architectures import baseline_parameter_count, match_baselines
from configuration_cfrd import CFRDConfig
from counterfactual_data import (
    RELATION_CATEGORIES,
    TRAIN_ENTITIES,
    VALIDATION_ENTITIES,
    CounterfactualPair,
    CounterfactualSampler,
)
from counterfactual_objective import counterfactual_ranking_result, encode_counterfactual_pairs
from evaluate import make_final_eval_rng
from model import CFRDLanguageModel, ModelConfig, count_parameters
from model_factory import (
    BASELINE_ARCH,
    CFRD_ARCH,
    architecture_of_checkpoint,
    architecture_of_config,
    build_model,
    build_model_config,
    model_config_from_checkpoint,
    normalize_architecture,
)
from modeling_cfrd import CFRDForCausalLM
from surface_features import SURFACE_FEATURE_DIM
from tokenization_cfrd import CFRDTokenizer
from train import configure_optimizer, restore_checkpoint, save_checkpoint

# Small synthetic setup that does not require a trained tokenizer.
TEST_VOCAB_SIZE = 512
TEST_CONTEXT_LENGTH = 32
TEST_CHUNK_SIZE = 8
TEST_D_MODEL = 64
TEST_N_HEAD = 4
TEST_N_KV_HEAD = 2
TEST_FFN_DIM = 128
TEST_MEMORY_DIM = 32
TEST_MEMORY_HEADS = 4
TEST_SUMMARY_SLOTS = 2
TEST_PHYSICAL_CELLS = 2
TEST_RECURRENCES = 4
TEST_EXIT_DEPTHS = (2, 4)


def build_test_model() -> CFRDLanguageModel:
    cfg = ModelConfig(
        vocab_size=TEST_VOCAB_SIZE,
        context_length=TEST_CONTEXT_LENGTH,
        chunk_size=TEST_CHUNK_SIZE,
        d_model=TEST_D_MODEL,
        n_head=TEST_N_HEAD,
        n_kv_head=TEST_N_KV_HEAD,
        ffn_dim=TEST_FFN_DIM,
        summary_slots=TEST_SUMMARY_SLOTS,
        memory_dim=TEST_MEMORY_DIM,
        memory_heads=TEST_MEMORY_HEADS,
        physical_cells=TEST_PHYSICAL_CELLS,
        recurrences=TEST_RECURRENCES,
        exit_depths=TEST_EXIT_DEPTHS,
        use_binding_block=True,
        use_surface_features=True,
        surface_feature_dim=SURFACE_FEATURE_DIM,
    )

    features = torch.randn(TEST_VOCAB_SIZE, SURFACE_FEATURE_DIM)
    model = CFRDLanguageModel(cfg, features)
    model.eval()
    return model


TEST_BASELINE_LAYERS = 3
TEST_BASELINE_FFN_DIM = 192


def build_test_baseline() -> BaselineLanguageModel:
    cfg = BaselineConfig(
        vocab_size=TEST_VOCAB_SIZE,
        context_length=TEST_CONTEXT_LENGTH,
        d_model=TEST_D_MODEL,
        n_head=TEST_N_HEAD,
        n_kv_head=TEST_N_KV_HEAD,
        ffn_dim=TEST_BASELINE_FFN_DIM,
        n_layer=TEST_BASELINE_LAYERS,
        use_surface_features=True,
        surface_feature_dim=SURFACE_FEATURE_DIM,
    )

    features = torch.randn(TEST_VOCAB_SIZE, SURFACE_FEATURE_DIM)
    model = BaselineLanguageModel(cfg, features)
    model.eval()
    return model


def test_shapes() -> None:
    model = build_test_model()
    x = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))
    y = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))

    output = model(x, targets=y)
    assert output.logits.shape == (2, TEST_CONTEXT_LENGTH, TEST_VOCAB_SIZE)
    assert output.loss is not None
    assert output.final_loss is not None
    assert set(output.exit_losses) == set(TEST_EXIT_DEPTHS)


def test_backward_pass() -> None:
    """Every trainable branch should receive a finite gradient."""

    torch.manual_seed(0)
    model = build_test_model()
    model.train()
    x = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))
    y = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))

    output = model(x, targets=y)
    assert output.loss is not None
    output.loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_causality_inside_chunk() -> None:
    """Changing a future token in one chunk must not alter earlier logits."""

    torch.manual_seed(1)
    model = build_test_model()

    x1 = torch.randint(0, TEST_VOCAB_SIZE, (1, TEST_CONTEXT_LENGTH))
    x2 = x1.clone()

    changed_position = 6
    x2[0, changed_position] = (x2[0, changed_position] + 1) % TEST_VOCAB_SIZE

    with torch.no_grad():
        y1 = model(x1).logits
        y2 = model(x2).logits

    # Earlier outputs must be independent of the changed future input.
    assert torch.allclose(y1[:, :changed_position], y2[:, :changed_position], atol=1.0e-6, rtol=1.0e-5)


def test_causality_across_chunks() -> None:
    """Changing future chunks must not alter logits in earlier chunks."""

    torch.manual_seed(2)
    model = build_test_model()

    x1 = torch.randint(0, TEST_VOCAB_SIZE, (1, TEST_CONTEXT_LENGTH))
    x2 = x1.clone()
    x2[:, 16:] = torch.randint(0, TEST_VOCAB_SIZE, x2[:, 16:].shape)

    with torch.no_grad():
        y1 = model(x1).logits
        y2 = model(x2).logits

    assert torch.allclose(y1[:, :16], y2[:, :16], atol=1.0e-6, rtol=1.0e-5)


def test_partial_chunk() -> None:
    """A sequence that ends mid-chunk should stay finite and preserve its length."""

    model = build_test_model()
    x = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH - 3))
    with torch.no_grad():
        output = model(x)
    assert output.logits.shape == (2, TEST_CONTEXT_LENGTH - 3, TEST_VOCAB_SIZE)
    assert torch.isfinite(output.logits).all()


def test_last_token_logits_match_full_projection() -> None:
    """Generation may skip unused vocabulary projections without changing scores."""

    model = build_test_model()
    x = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))
    with torch.no_grad():
        full_logits = model(x).logits
        last_logits = model(x, logits_to_keep=1).logits

    assert last_logits.shape == (2, 1, TEST_VOCAB_SIZE)
    assert torch.equal(last_logits, full_logits[:, -1:, :])


def test_binding_block_config_compatibility() -> None:
    """Old checkpoints stay disabled while new HF configs preserve the block."""

    legacy_checkpoint = {
        "model_config": {
            "vocab_size": TEST_VOCAB_SIZE,
            "context_length": TEST_CONTEXT_LENGTH,
            "chunk_size": TEST_CHUNK_SIZE,
            "d_model": TEST_D_MODEL,
            "n_head": TEST_N_HEAD,
            "n_kv_head": TEST_N_KV_HEAD,
            "ffn_dim": TEST_FFN_DIM,
            "summary_slots": TEST_SUMMARY_SLOTS,
            "memory_dim": TEST_MEMORY_DIM,
            "memory_heads": TEST_MEMORY_HEADS,
            "physical_cells": TEST_PHYSICAL_CELLS,
            "recurrences": TEST_RECURRENCES,
            "exit_depths": list(TEST_EXIT_DEPTHS),
        }
    }
    restored = ModelConfig.from_checkpoint(legacy_checkpoint, TEST_VOCAB_SIZE)
    assert not restored.use_binding_block

    enabled = build_test_model().cfg
    hf_config = CFRDConfig.from_model_config(enabled)
    assert hf_config.use_binding_block
    assert hf_config.tie_word_embeddings
    assert hf_config.to_model_config().use_binding_block


def test_binding_block_supervises_every_exit() -> None:
    """Training and terminal inference must project through the same binding path."""

    model = build_test_model()
    assert model.binding_block is not None
    x = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))
    y = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))
    binding_calls = 0

    def count_binding_calls(_module, _args, _output) -> None:
        nonlocal binding_calls
        binding_calls += 1

    handle = model.binding_block.register_forward_hook(count_binding_calls)
    try:
        model(x, targets=y)
        assert binding_calls == len(TEST_EXIT_DEPTHS)

        binding_calls = 0
        model(x, recurrences=TEST_EXIT_DEPTHS[0])
        assert binding_calls == 1
    finally:
        handle.remove()


def test_final_depth_evaluation_reuses_windows() -> None:
    """Every depth comparison must draw the same ordered validation windows."""

    first_starts = make_final_eval_rng().integers(0, 10_000, size=(8, 4))
    second_starts = make_final_eval_rng().integers(0, 10_000, size=(8, 4))
    assert np.array_equal(first_starts, second_starts)


def test_exact_sampler_resume() -> None:
    """Checkpoint restore must continue both independent sampler sequences."""

    model = build_test_model()
    optimizer = configure_optimizer(model, torch.device("cpu"))
    lm_rng = np.random.default_rng(123)
    counterfactual_rng = np.random.default_rng(456)
    lm_rng.integers(0, 10_000, size=100)
    counterfactual_rng.integers(0, 10_000, size=50)

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "checkpoint.pt"
        save_checkpoint(
            path,
            model,
            optimizer,
            7,
            1234,
            2.0,
            model.cfg,
            "test-hash",
            lm_rng,
            counterfactual_rng,
        )
        expected_pairs = CounterfactualSampler("train").sample_batch(5, counterfactual_rng)
        expected = lm_rng.integers(0, 10_000, size=20)

        restored_model = build_test_model()
        restored_optimizer = configure_optimizer(restored_model, torch.device("cpu"))
        restored_lm_rng = np.random.default_rng(999)
        restored_counterfactual_rng = np.random.default_rng(888)
        step, tokens_seen, best_loss, rng_mode = restore_checkpoint(
            path,
            restored_model,
            restored_optimizer,
            model.cfg,
            "test-hash",
            restored_lm_rng,
            restored_counterfactual_rng,
        )
        actual_pairs = CounterfactualSampler("train").sample_batch(5, restored_counterfactual_rng)
        actual = restored_lm_rng.integers(0, 10_000, size=20)

    assert step == 7
    assert tokens_seen == 1234
    assert best_loss == 2.0
    assert rng_mode == "separate"
    assert actual_pairs == expected_pairs
    assert np.array_equal(actual, expected)


def test_legacy_shared_sampler_resume() -> None:
    """Version-3 checkpoints must preserve their historical shared RNG sequence."""

    model = build_test_model()
    optimizer = configure_optimizer(model, torch.device("cpu"))
    shared_rng = np.random.default_rng(321)
    shared_rng.integers(0, 10_000, size=100)

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "checkpoint.pt"
        save_checkpoint(path, model, optimizer, 7, 1234, 2.0, model.cfg, "test-hash", shared_rng, shared_rng)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["checkpoint_version"] = 3
        checkpoint.pop("training_rng_mode")
        checkpoint["train_rng_state"] = checkpoint.pop("lm_rng_state")
        checkpoint.pop("counterfactual_rng_state")
        checkpoint["optimizer_config"].pop("counterfactual_seed_offset")
        torch.save(checkpoint, path)

        expected_pairs = CounterfactualSampler("train").sample_batch(5, shared_rng)
        expected = shared_rng.integers(0, 10_000, size=20)

        restored_model = build_test_model()
        restored_optimizer = configure_optimizer(restored_model, torch.device("cpu"))
        restored_lm_rng = np.random.default_rng(999)
        unused_counterfactual_rng = np.random.default_rng(888)
        _, _, _, rng_mode = restore_checkpoint(
            path,
            restored_model,
            restored_optimizer,
            model.cfg,
            "test-hash",
            restored_lm_rng,
            unused_counterfactual_rng,
        )
        actual_pairs = CounterfactualSampler("train").sample_batch(5, restored_lm_rng)
        actual = restored_lm_rng.integers(0, 10_000, size=20)

    assert rng_mode == "shared"
    assert actual_pairs == expected_pairs
    assert np.array_equal(actual, expected)


def test_transformers_auto_model_roundtrip() -> None:
    """Safetensors and AutoModel must preserve logits and generation support."""

    transformers_logging.disable_progress_bar()
    core_model = build_test_model()
    hf_config = CFRDConfig.from_model_config(core_model.cfg, inference_recurrences=4)
    hf_model = CFRDForCausalLM(hf_config).eval()
    hf_model.model.load_state_dict(core_model.state_dict())
    assert hf_model.get_input_embeddings() is hf_model.get_output_embeddings()
    input_ids = torch.randint(0, TEST_VOCAB_SIZE, (1, 12))

    with torch.no_grad():
        expected_logits = hf_model(input_ids).logits

    with tempfile.TemporaryDirectory() as temp_dir:
        hf_model.save_pretrained(temp_dir, safe_serialization=True)
        loaded = AutoModelForCausalLM.from_pretrained(
            temp_dir,
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        with torch.no_grad():
            actual_logits = loaded(input_ids).logits
            generated = loaded.generate(
                input_ids=input_ids[:, :3],
                max_new_tokens=2,
                do_sample=False,
                use_cache=False,
                recurrences=2,
            )

    assert loaded.can_generate()
    assert generated.shape == (1, 5)
    assert torch.equal(expected_logits, actual_logits)


def test_transformers_tokenizer_roundtrip() -> None:
    """AutoTokenizer must preserve Korean text, newlines, and literal control text."""

    sentences = [
        "작은 마을에 아침이 왔어요.",
        "literal <|nl|> marker\nand a real newline",
    ] * 20

    with tempfile.TemporaryDirectory() as temp_dir:
        model_prefix = str(Path(temp_dir) / "test_tokenizer")
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(sentences),
            model_prefix=model_prefix,
            vocab_size=300,
            model_type="unigram",
            character_coverage=1.0,
            byte_fallback=True,
            hard_vocab_limit=False,
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=["<|nl|>", "<|literal_nl|>"],
            minloglevel=2,
        )
        tokenizer = CFRDTokenizer(
            vocab_file=model_prefix + ".model",
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
        )
        export_directory = Path(temp_dir) / "export"
        tokenizer.save_pretrained(export_directory)
        loaded = AutoTokenizer.from_pretrained(
            export_directory,
            trust_remote_code=True,
            local_files_only=True,
        )

        sample = "literal <|nl|> marker\nand a real newline"
        expected_ids = tokenizer(sample).input_ids
        actual_ids = loaded(sample).input_ids
        decoded = loaded.decode(actual_ids, skip_special_tokens=True)

    assert actual_ids == expected_ids
    assert decoded == sample


def test_baseline_shapes_and_exit() -> None:
    """The baseline must satisfy the ModelOutput contract shared code relies on."""

    model = build_test_baseline()
    x = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))
    y = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))

    output = model(x, targets=y)
    assert output.logits.shape == (2, TEST_CONTEXT_LENGTH, TEST_VOCAB_SIZE)
    assert output.loss is not None
    assert output.final_loss is not None
    # train.py preallocates its running totals from cfg.exit_depths, so the
    # single dense exit has to appear under exactly that key.
    assert model.cfg.exit_depths == (TEST_BASELINE_LAYERS,)
    assert set(output.exit_losses) == {TEST_BASELINE_LAYERS}
    assert torch.equal(output.loss, output.final_loss)


def test_baseline_backward_pass() -> None:
    torch.manual_seed(0)
    model = build_test_baseline()
    model.train()
    x = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))
    y = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))

    output = model(x, targets=y)
    assert output.loss is not None
    output.loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_baseline_causality() -> None:
    """A control with broken causality would understate CFRD, not just differ."""

    torch.manual_seed(3)
    model = build_test_baseline()

    x1 = torch.randint(0, TEST_VOCAB_SIZE, (1, TEST_CONTEXT_LENGTH))
    x2 = x1.clone()
    x2[:, 16:] = torch.randint(0, TEST_VOCAB_SIZE, x2[:, 16:].shape)

    with torch.no_grad():
        y1 = model(x1).logits
        y2 = model(x2).logits

    assert torch.allclose(y1[:, :16], y2[:, :16], atol=1.0e-6, rtol=1.0e-5)


def test_baseline_last_token_logits_match_full_projection() -> None:
    model = build_test_baseline()
    x = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))
    with torch.no_grad():
        full_logits = model(x).logits
        last_logits = model(x, logits_to_keep=1).logits

    assert last_logits.shape == (2, 1, TEST_VOCAB_SIZE)
    assert torch.equal(last_logits, full_logits[:, -1:, :])


def test_baseline_rejects_recurrent_depth() -> None:
    """A dense stack must not answer a recurrent-depth sweep with one number.

    evaluate.py loops over cfg.exit_depths and passes each as `recurrences`.
    Accepting and ignoring a shallower depth would emit three identical rows
    that look like a depth comparison.
    """

    model = build_test_baseline()
    x = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))

    with torch.no_grad():
        assert model(x, recurrences=None).logits.shape[1] == TEST_CONTEXT_LENGTH
        assert model(x, recurrences=TEST_BASELINE_LAYERS).logits.shape[1] == TEST_CONTEXT_LENGTH

    for depth in (1, TEST_BASELINE_LAYERS - 1, TEST_BASELINE_LAYERS + 1):
        try:
            model(x, recurrences=depth)
        except ValueError:
            continue
        raise AssertionError(f"The baseline silently accepted recurrences={depth}")


def test_model_factory_dispatch() -> None:
    """config.MODEL_ARCH must fully determine which model a run builds."""

    settings = SimpleNamespace(**{name: getattr(config, name) for name in dir(config) if name.isupper()})
    features = torch.randn(TEST_VOCAB_SIZE, SURFACE_FEATURE_DIM)

    for architecture, expected_type in ((CFRD_ARCH, CFRDLanguageModel), (BASELINE_ARCH, BaselineLanguageModel)):
        settings.MODEL_ARCH = architecture
        model_cfg = build_model_config(settings, TEST_VOCAB_SIZE)
        assert architecture_of_config(model_cfg) == architecture

        model = build_model(model_cfg, features)
        assert isinstance(model, expected_type)

        checkpoint = {"model_arch": architecture, "model_config": asdict(model_cfg)}
        assert architecture_of_checkpoint(checkpoint) == architecture
        assert model_config_from_checkpoint(checkpoint, TEST_VOCAB_SIZE) == model_cfg

    try:
        normalize_architecture("transformer")
    except ValueError:
        pass
    else:
        raise AssertionError("An unknown architecture name was accepted")


def test_untagged_checkpoint_reads_as_cfrd() -> None:
    """Released v1.0 and v1.1 checkpoints predate the model_arch tag."""

    legacy = {"model_config": asdict(build_test_model().cfg)}
    assert "model_arch" not in legacy
    assert architecture_of_checkpoint(legacy) == CFRD_ARCH

    restored = model_config_from_checkpoint(legacy, TEST_VOCAB_SIZE)
    assert isinstance(restored, ModelConfig)
    assert restored == build_test_model().cfg


def test_baseline_checkpoint_resume_and_arch_guard() -> None:
    """The baseline needs exact resume, and must never load CFRD weights."""

    torch.manual_seed(7)
    device = torch.device("cpu")
    model = build_test_baseline()
    model.train()
    optimizer = configure_optimizer(model, device)

    x = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))
    y = torch.randint(0, TEST_VOCAB_SIZE, (2, TEST_CONTEXT_LENGTH))
    output = model(x, targets=y)
    assert output.loss is not None
    output.loss.backward()
    optimizer.step()

    lm_rng = np.random.default_rng(11)
    counterfactual_rng = np.random.default_rng(12)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "baseline.pt"
        save_checkpoint(path, model, optimizer, 5, 640, 1.25, model.cfg, "hash", lm_rng, counterfactual_rng)

        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["checkpoint_version"] == 5
        assert saved["model_arch"] == BASELINE_ARCH

        restored_model = build_test_baseline()
        restored_model.train()
        restored_optimizer = configure_optimizer(restored_model, device)
        step, tokens_seen, best_val_loss, _ = restore_checkpoint(
            path,
            restored_model,
            restored_optimizer,
            model.cfg,
            "hash",
            np.random.default_rng(0),
            np.random.default_rng(0),
        )

        assert (step, tokens_seen, best_val_loss) == (5, 640, 1.25)
        for original, restored in zip(model.parameters(), restored_model.parameters()):
            assert torch.equal(original, restored)

        # Loading a baseline checkpoint into CFRD must fail on the architecture,
        # not on an unreadable list of non-overlapping config fields.
        cfrd_model = build_test_model()
        try:
            restore_checkpoint(
                path,
                cfrd_model,
                configure_optimizer(cfrd_model, device),
                cfrd_model.cfg,
                "hash",
                np.random.default_rng(0),
                np.random.default_rng(0),
            )
        except ValueError as error:
            assert BASELINE_ARCH in str(error) and CFRD_ARCH in str(error)
        else:
            raise AssertionError("A baseline checkpoint was loaded into CFRD")


def test_matched_sizing_is_exact() -> None:
    """The closed-form sizing must agree with the models it claims to match.

    compare_architectures.py sizes baselines analytically instead of building a
    candidate per trial. If the formula drifted from BaselineLanguageModel, the
    comparison would silently report a mismatched control as matched.
    """

    cfrd_cfg = build_test_model().cfg
    features = torch.randn(TEST_VOCAB_SIZE, SURFACE_FEATURE_DIM)
    cfrd_parameters = count_parameters(build_model(cfrd_cfg, features))["total"]

    for name, baseline_cfg in match_baselines(cfrd_cfg, cfrd_parameters).items():
        predicted = baseline_parameter_count(
            vocab_size=baseline_cfg.vocab_size,
            d_model=baseline_cfg.d_model,
            n_head=baseline_cfg.n_head,
            n_kv_head=baseline_cfg.n_kv_head,
            ffn=baseline_cfg.ffn_dim,
            layers=baseline_cfg.n_layer,
        )
        actual = count_parameters(build_model(baseline_cfg, features))["total"]
        assert predicted == actual, f"{name}: predicted {predicted:,} but built {actual:,}"

        # Both baselines inherit everything CFRD does not vary, so a difference
        # in results cannot be blamed on context, heads, or the embedding path.
        assert baseline_cfg.context_length == cfrd_cfg.context_length
        assert (baseline_cfg.d_model, baseline_cfg.n_head, baseline_cfg.n_kv_head) == (
            cfrd_cfg.d_model,
            cfrd_cfg.n_head,
            cfrd_cfg.n_kv_head,
        )
        assert baseline_cfg.use_surface_features == cfrd_cfg.use_surface_features

    matched = match_baselines(cfrd_cfg, cfrd_parameters)
    parameter_matched = matched["baseline-param-matched"]
    compute_matched = matched["baseline-compute-matched"]

    # The parameter-matched arm must actually hold parameters.
    parameter_error = (
        abs(count_parameters(build_model(parameter_matched, features))["total"] - cfrd_parameters) / cfrd_parameters
    )
    assert parameter_error < 0.02, f"parameter match drifted to {parameter_error:.2%}"

    # The compute-matched arm must actually hold sequential depth.
    expected_depth = cfrd_cfg.recurrences + (1 if cfrd_cfg.use_binding_block else 0)
    assert compute_matched.n_layer == expected_depth
    assert compute_matched.ffn_dim == parameter_matched.ffn_dim


def print_default_parameter_count() -> None:
    cfg = ModelConfig.from_project_settings(config, config.TOKENIZER_VOCAB_SIZE)
    dummy_features = torch.zeros(config.TOKENIZER_VOCAB_SIZE, SURFACE_FEATURE_DIM)
    model = CFRDLanguageModel(cfg, dummy_features)
    counts = count_parameters(model)
    assert counts["total"] == 11_634_459
    assert 8_000_000 <= counts["total"] <= 12_000_000
    print(f"default total parameters: {counts['total']:,}")


def test_counterfactual_sampler() -> None:
    first_rng = np.random.default_rng(20260812)
    second_rng = np.random.default_rng(20260812)
    train_sampler = CounterfactualSampler("train")

    first_batch = train_sampler.sample_batch(15, first_rng)
    repeated_batch = train_sampler.sample_batch(15, second_rng)
    next_batch = train_sampler.sample_batch(15, first_rng)

    assert first_batch == repeated_batch
    assert first_batch != next_batch
    assert {pair.category for pair in first_batch} == set(RELATION_CATEGORIES)
    assert len({pair.entity_a for pair in first_batch[:12]} | {pair.entity_b for pair in first_batch[:12]}) == 24

    for pair in first_batch:
        assert pair.entity_a != pair.entity_b
        assert pair.prompt_a != pair.prompt_b
        assert pair.candidates[0] != pair.candidates[1]
        assert pair.correct_indices == (0, 1)

    validation_batch = CounterfactualSampler("validation").sample_batch(10, np.random.default_rng(7))
    assert set(TRAIN_ENTITIES).isdisjoint(VALIDATION_ENTITIES)
    assert {pair.template_id for pair in first_batch}.isdisjoint(pair.template_id for pair in validation_batch)


def test_counterfactual_objective_backward() -> None:
    class CharacterTokenizer:
        pad_id = 0

        def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
            ids = [4 + (ord(character) % (TEST_VOCAB_SIZE - 4)) for character in text]
            return ([2] if add_bos else []) + ids + ([3] if add_eos else [])

    pair = CounterfactualPair(
        category="test",
        template_id="test:0",
        entity_a="A",
        entity_b="B",
        prompt_a="A=x; B=y; A?",
        prompt_b="A=y; B=x; A?",
        candidates=("x", "y"),
    )
    model = build_test_model()
    batch = encode_counterfactual_pairs(
        (pair,),
        CharacterTokenizer(),
        context_length=TEST_CONTEXT_LENGTH,
        device=torch.device("cpu"),
    )
    result = counterfactual_ranking_result(model, batch, margin=0.5)
    result.loss.backward()

    assert batch.input_ids.shape[0] == 4
    assert batch.score_mask.sum().item() == 4
    assert torch.isfinite(result.loss)
    assert 0.0 <= result.decision_accuracy.item() <= 1.0
    assert 0.0 <= result.strict_pair_accuracy.item() <= 1.0
    assert model.token_embedding.weight.grad is not None


def main() -> None:
    test_shapes()
    test_backward_pass()
    test_causality_inside_chunk()
    test_causality_across_chunks()
    test_partial_chunk()
    test_last_token_logits_match_full_projection()
    test_binding_block_config_compatibility()
    test_binding_block_supervises_every_exit()
    test_final_depth_evaluation_reuses_windows()
    test_exact_sampler_resume()
    test_legacy_shared_sampler_resume()
    test_transformers_auto_model_roundtrip()
    test_counterfactual_sampler()
    test_counterfactual_objective_backward()
    test_transformers_tokenizer_roundtrip()
    test_baseline_shapes_and_exit()
    test_baseline_backward_pass()
    test_baseline_causality()
    test_baseline_last_token_logits_match_full_projection()
    test_baseline_rejects_recurrent_depth()
    test_model_factory_dispatch()
    test_untagged_checkpoint_reads_as_cfrd()
    test_baseline_checkpoint_resume_and_arch_guard()
    test_matched_sizing_is_exact()
    print_default_parameter_count()
    print("smoke tests passed")


if __name__ == "__main__":
    main()
