from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from nl2sql_rl.training.sft import SFTConfig, inject_lora  # noqa: E402


@pytest.mark.sft_runtime
def test_tiny_random_model_lora_injection_save_and_merge(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model_config = transformers.GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=1,
    )
    base = transformers.GPT2LMHeadModel(model_config)
    config = SFTConfig(
        train_data=tmp_path / "train.jsonl",
        validation_data=tmp_path / "validation.jsonl",
        output_dir=tmp_path / "output",
    ).model_copy(update={"target_modules": ["c_attn"]})
    lora_model = inject_lora(base, config)
    assert any("lora_" in name for name, _ in lora_model.named_parameters())
    input_ids = torch.randint(0, 64, (1, 8))
    assert lora_model(input_ids=input_ids).logits.shape == (1, 8, 64)

    adapter = tmp_path / "adapter"
    lora_model.save_pretrained(adapter)
    fresh_base = transformers.GPT2LMHeadModel(model_config)
    loaded = peft.PeftModel.from_pretrained(fresh_base, adapter)
    merged = loaded.merge_and_unload()
    assert not any("lora_" in name for name, _ in merged.named_parameters())
    assert merged(input_ids=input_ids).logits.shape == (1, 8, 64)
