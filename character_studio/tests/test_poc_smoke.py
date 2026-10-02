"""Runs the whole proof-of-concept flow with a tiny random-weight FLUX.2 [klein]
model saved to disk and loaded through the real loading code, on the CPU.
The pictures are noise; this checks the plumbing (memory modes, staged
loading, mixed-precision streaming, prompts, saving), not picture quality.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
from PIL import Image  # noqa: E402

import poc  # noqa: E402
from studio import engine as engine_mod  # noqa: E402
from studio.engine import Engine, choose_mode, prepare_reference  # noqa: E402


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


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """Tiny model saved in bfloat16, like the real download."""
    path = tmp_path_factory.mktemp("tiny_klein")
    tiny_pipeline().to(torch.bfloat16).save_pretrained(path)
    return str(path)


class SmallEngine(Engine):
    """Real engine with tiny picture sizes and 2 steps so the CPU test is quick."""

    def _run(self, prompt, references, seed, width, height, steps):
        return super()._run(prompt, references, seed, 64, 96, 2)


@pytest.fixture(params=["staged", "gpu"])
def engine(request, tiny_model_dir, monkeypatch):
    monkeypatch.setattr(engine_mod, "MODEL_ID", tiny_model_dir)
    # staged: bf16 weights streamed with float32 maths, as on a GTX 10xx card
    dtype = torch.float32 if request.param == "staged" else torch.bfloat16
    return SmallEngine(log=lambda *a: None, mode=request.param, device="cpu", compute_dtype=dtype)


def test_mode_choice():
    assert choose_mode(24, True, 64) == "gpu"
    assert choose_mode(16, True, 32) == "offload"
    assert choose_mode(12, True, 32) == "offload"
    assert choose_mode(12, True, 16) == "staged"      # not enough RAM to hold both parts
    assert choose_mode(8, False, 16) == "staged"      # GTX 1070 + 16 GB
    assert choose_mode(24, False, 64) == "staged"     # big but old card


def test_prepare_reference_flattens_transparency():
    img = Image.new("RGBA", (2000, 1000), (0, 0, 0, 0))
    out = prepare_reference(img)
    assert out.mode == "RGB" and max(out.size) == 1024
    assert out.getpixel((0, 0)) == (234, 244, 74)


def test_staged_matches_full_pipeline_text_encoding(tiny_model_dir, monkeypatch):
    """Our own staged text encoding must give what the official pipeline gives."""
    from diffusers import Flux2KleinPipeline
    monkeypatch.setattr(engine_mod, "MODEL_ID", tiny_model_dir)
    pipe = Flux2KleinPipeline.from_pretrained(tiny_model_dir, dtype=torch.float32)
    expected = pipe.encode_prompt("a pig girl, blue hair", device="cpu")[0]
    eng = Engine(log=lambda *a: None, mode="staged", device="cpu", compute_dtype=torch.float32)
    eng.prepare(["a pig girl, blue hair"])
    got = eng._embeds["a pig girl, blue hair"]
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-2, rtol=1e-2)


def test_same_seed_is_repeatable(engine):
    a = engine.create("a pig", seed=3)
    b = engine.create("a pig", seed=3)
    assert a.size == (64, 96)
    assert a.tobytes() == b.tobytes()


def test_staged_loads_one_part_at_a_time(engine):
    if engine.mode != "staged":
        pytest.skip("staged only")
    engine.create("a pig", seed=1)
    assert engine.pipe is not None and engine.pipe.text_encoder is None
    engine.prepare(["a new description"])   # needs the text model again -> image model unloaded
    assert engine.pipe is None
    engine.edit("a new description", [Image.new("RGB", (64, 80), "yellow")], seed=1)
    assert engine.pipe is not None


def test_poc_from_description(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(poc, "HERE", str(tmp_path))
    monkeypatch.setattr(poc, "Engine", lambda: engine)
    monkeypatch.setattr(poc, "remove_background", lambda img: img.convert("RGBA"))
    assert poc.main(["--describe", "a pig girl with blue hair", "--poses", "waving", "back",
                     "--no-open"]) == 0
    (folder,) = (tmp_path / "output").iterdir()
    names = sorted(os.listdir(folder))
    assert names == ["00_character.png", "00_character_transparent.png", "01_waving.png",
                     "01_waving_transparent.png", "02_back.png", "02_back_transparent.png",
                     "character.json", "comparison_sheet.png"]
    meta = json.loads((folder / "character.json").read_text())
    assert meta["poses"] == ["waving", "back"]


def test_poc_from_reference(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(poc, "HERE", str(tmp_path))
    monkeypatch.setattr(poc, "Engine", lambda: engine)
    monkeypatch.setattr(poc, "remove_background", lambda img: None)
    ref = tmp_path / "Leah.png"
    Image.new("RGB", (64, 80), (234, 244, 74)).save(ref)
    assert poc.main(["--reference", str(ref), "--poses", "sitting on a bench reading",
                     "--no-open"]) == 0
    (folder,) = (tmp_path / "output").iterdir()
    assert folder.name.startswith("Leah_")
    assert {"00_reference.png", "01_sitting_on_a_bench_reading.png",
            "comparison_sheet.png"} <= set(os.listdir(folder))
