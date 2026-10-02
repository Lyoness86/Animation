"""Runs the whole proof-of-concept flow with a tiny random-weight FLUX.2 [klein]
pipeline on the CPU. The pictures are noise; this checks the plumbing
(prompts, reference handling, saving, contact sheet), not picture quality.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
from PIL import Image  # noqa: E402

import poc  # noqa: E402
from studio.engine import Engine, pick_memory_mode, prepare_reference  # noqa: E402


def tiny_pipeline():
    from diffusers import (AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler,
                           Flux2KleinPipeline, Flux2Transformer2DModel)
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    vocab = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2}
    for ch in sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,()-'\n")):
        vocab.setdefault(ch, len(vocab))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<|endoftext|>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tokenizer = Qwen2TokenizerFast(tokenizer_object=tok, pad_token="<|endoftext|>",
                                   eos_token="<|im_end|>", unk_token="<|endoftext|>")
    tokenizer.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
                               "<|im_end|>\n{% endfor %}<|im_start|>assistant\n")
    text_encoder = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=28,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=1024))
    transformer = Flux2Transformer2DModel(
        in_channels=16, num_layers=1, num_single_layers=1, attention_head_dim=16,
        num_attention_heads=2, joint_attention_dim=48, timestep_guidance_channels=32,
        axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False)
    vae = AutoencoderKLFlux2(block_out_channels=(8, 8), down_block_types=("DownEncoderBlock2D",) * 2,
                             up_block_types=("UpDecoderBlock2D",) * 2, layers_per_block=1,
                             latent_channels=4, norm_num_groups=4, sample_size=32)
    return Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae,
                              text_encoder=text_encoder, tokenizer=tokenizer,
                              transformer=transformer, is_distilled=True)


@pytest.fixture
def tiny_engine(monkeypatch):
    pipe = tiny_pipeline()
    pipe.set_progress_bar_config(disable=True)
    eng = Engine(pipe=pipe, device="cpu", log=lambda *a: None)
    # small pictures and few steps keep this fast on the CPU
    orig_run = eng._run
    eng._run = lambda prompt, refs, seed, w, h, steps: orig_run(prompt, refs, seed, 64, 96, 2)
    monkeypatch.setattr(poc, "Engine", lambda: eng)
    monkeypatch.setattr(poc, "remove_background", lambda img: img.convert("RGBA"))
    return eng


def test_memory_mode_thresholds():
    assert pick_memory_mode(24) == "gpu"
    assert pick_memory_mode(16) == "model_offload"
    assert pick_memory_mode(12) == "model_offload"
    assert pick_memory_mode(8) == "sequential_offload"


def test_prepare_reference_flattens_transparency():
    img = Image.new("RGBA", (2000, 1000), (0, 0, 0, 0))
    out = prepare_reference(img)
    assert out.mode == "RGB" and max(out.size) == 1024
    assert out.getpixel((0, 0)) == (234, 244, 74)


def test_same_seed_is_repeatable(tiny_engine):
    a = tiny_engine.create("a pig", seed=3)
    b = tiny_engine.create("a pig", seed=3)
    assert a.tobytes() == b.tobytes()


def test_poc_from_description(tiny_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(poc, "HERE", str(tmp_path))
    assert poc.main(["--describe", "a pig girl with blue hair", "--poses", "waving", "back",
                     "--no-open"]) == 0
    (folder,) = (tmp_path / "output").iterdir()
    names = sorted(os.listdir(folder))
    assert names == ["00_character.png", "00_character_transparent.png", "01_waving.png",
                     "01_waving_transparent.png", "02_back.png", "02_back_transparent.png",
                     "character.json", "comparison_sheet.png"]
    meta = json.loads((folder / "character.json").read_text())
    assert meta["poses"] == ["waving", "back"]


def test_poc_from_reference(tiny_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(poc, "HERE", str(tmp_path))
    ref = tmp_path / "Leah.png"
    Image.new("RGB", (300, 400), (234, 244, 74)).save(ref)
    assert poc.main(["--reference", str(ref), "--poses", "sitting on a bench reading",
                     "--no-open"]) == 0
    (folder,) = (tmp_path / "output").iterdir()
    assert folder.name.startswith("Leah_")
    assert {"00_reference.png", "01_sitting_on_a_bench_reading.png",
            "comparison_sheet.png"} <= set(os.listdir(folder))
