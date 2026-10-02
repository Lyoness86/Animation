"""Local image generation with FLUX.2 [klein] 4B (runs entirely on this PC).

One model does both jobs:
  * text -> new character
  * reference picture(s) + text -> the same character in a new pose/angle

The model has two big parts: a text encoder (~8 GB) that turns the words into
numbers, and the image model (~8 GB) that draws. On big PCs both stay loaded.
On smaller PCs ("staged" mode) only one part is in memory at a time: all
prompts are encoded first, the text encoder is unloaded, then the image model
is loaded and streamed through the graphics card block by block.
"""
import gc
import os
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Keep the downloaded model inside this folder so it is easy to find and delete.
os.environ.setdefault("HF_HOME", os.path.join(HERE, "models"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("U2NET_HOME", os.path.join(HERE, "models", "background_removal"))

from PIL import Image  # noqa: E402

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
STEPS = 4            # the model is distilled for 4 steps
WIDTH, HEIGHT = 768, 1024   # portrait, suits full-body characters
TEXT_LAYERS = (9, 18, 27)   # text-encoder layers the image model reads (same as the pipeline)


def hardware():
    """(gpu name, vram GB, bf16 capable, system RAM GB). gpu name is None without a usable GPU.

    AMD cards show up through the same torch.cuda calls when AMD's ROCm PyTorch is installed.
    """
    import psutil
    import torch
    ram = psutil.virtual_memory().total / 1024 ** 3
    if not torch.cuda.is_available():
        return None, 0.0, False, ram
    props = torch.cuda.get_device_properties(0)
    if torch.version.hip:   # AMD (ROCm): PyTorch knows whether the card handles bfloat16
        bf16 = torch.cuda.is_bf16_supported()
    else:                   # NVIDIA before RTX 30xx (GTX 10xx, RTX 20xx) can't do bfloat16 maths
        bf16 = props.major >= 8
    return props.name, props.total_memory / 1024 ** 3, bf16, ram


def choose_mode(vram_gb, bf16_ok, ram_gb):
    """gpu: everything on the card. offload: whole parts swapped in/out. staged: see module doc."""
    if not bf16_ok or vram_gb < 11.5 or ram_gb < 24:
        return "staged"
    if vram_gb >= 22:
        return "gpu"
    return "offload"


def stream_through_gpu(module, device, compute_dtype, skip=("norm",)):
    """Keep `module`'s weights in RAM in their stored precision; on each use, copy one block at
    a time to `device` and compute in `compute_dtype` (float32 on older cards)."""
    import torch
    from diffusers.hooks import apply_group_offloading, apply_layerwise_casting

    storage = next(module.parameters()).dtype
    if compute_dtype != storage:
        apply_layerwise_casting(module, storage_dtype=storage, compute_dtype=compute_dtype,
                                skip_modules_pattern=tuple(skip))
        # whatever the casting skipped (norms etc., all small) must already be in compute dtype
        for sub in module.modules():
            registry = getattr(sub, "_diffusers_hook", None)
            if registry is None or registry.get_hook("layerwise_casting") is None:
                for p in sub.parameters(recurse=False):
                    p.data = p.data.to(compute_dtype)
    if torch.device(device).type != "cpu":
        apply_group_offloading(module, onload_device=torch.device(device), offload_type="block_level",
                               num_blocks_per_group=1)


def free_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Engine:
    def __init__(self, log=print, mode=None, device=None, compute_dtype=None):
        import torch
        self.log = log
        self.pipe = None          # image pipeline (in gpu/offload modes it also holds the text encoder)
        self.tokenizer = None
        self.text_encoder = None  # only used separately in staged mode
        self._embeds = {}         # prompt -> encoded prompt (kept in RAM)
        if device is None:
            name, vram, bf16, ram = hardware()
            if name is None:
                raise RuntimeError(
                    "No usable graphics card is available to Python. Run check_my_pc.bat; "
                    "if it shows a suitable card, update its driver and run setup.bat again.")
            device = "cuda"
            mode = mode or choose_mode(vram, bf16, ram)
            compute_dtype = compute_dtype or (torch.bfloat16 if bf16 else torch.float32)
            self.log(f"Graphics card: {name} ({vram:.0f} GB), RAM {ram:.0f} GB - mode: {mode}"
                     f"{'' if bf16 else ', older card: computing in full precision'}")
        self.device = device
        self.mode = mode or "staged"
        self.compute_dtype = compute_dtype or torch.float32
        if self.mode != "staged":
            self._load_full()

    # ---------- loading ----------
    def _load_full(self):
        import torch
        from diffusers import Flux2KleinPipeline
        self.log("Loading the model (the first time this downloads about 16 GB)...")
        t = time.time()
        pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        if self.mode == "gpu":
            pipe.to(self.device)
        else:
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.log(f"Model ready in {time.time() - t:.0f} s")

    def _load_text_parts(self):
        """(tokenizer, text encoder) for staged mode."""
        import torch
        from transformers import AutoTokenizer, Qwen3ForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        text_encoder = Qwen3ForCausalLM.from_pretrained(MODEL_ID, subfolder="text_encoder",
                                                        dtype=torch.bfloat16)
        return tokenizer, text_encoder

    def _load_image_pipe(self):
        """Image pipeline without text encoder, for staged mode."""
        import torch
        from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
        transformer = Flux2Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer",
                                                              dtype=torch.bfloat16)
        vae = AutoencoderKLFlux2.from_pretrained(MODEL_ID, subfolder="vae", dtype=torch.float32)
        return Flux2KleinPipeline.from_pretrained(MODEL_ID, transformer=transformer, vae=vae,
                                                  text_encoder=None, tokenizer=None)

    def _ensure_image_model(self):
        if self.pipe is not None:
            return
        from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN
        self.log("Loading the image model...")
        t = time.time()
        pipe = self._load_image_pipe()
        skip = DEFAULT_SKIP_MODULES_PATTERN + tuple(
            getattr(pipe.transformer, "_skip_layerwise_casting_patterns", None) or ())
        stream_through_gpu(pipe.transformer, self.device, self.compute_dtype, skip)
        pipe.vae.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        free_memory()
        self.log(f"  ready in {time.time() - t:.0f} s")

    # ---------- text ----------
    def prepare(self, prompts):
        """Encode every prompt that will be needed. In staged mode this is done in one go so the
        text encoder only has to be loaded once."""
        todo = [p for p in dict.fromkeys(prompts) if p not in self._embeds]
        if not todo:
            return
        if self.mode != "staged":
            for p in todo:
                self._embeds[p] = self.pipe.encode_prompt(p, text_encoder_out_layers=TEXT_LAYERS)[0].cpu()
            return
        if self.pipe is not None:      # make room: the two parts don't fit in memory together
            self.pipe = None
            free_memory()
        self.log(f"Reading {len(todo)} description(s) with the text model...")
        t = time.time()
        tokenizer, text_encoder = self._load_text_parts()
        body = text_encoder.model      # the hidden states are all we need, not word predictions
        stream_through_gpu(body, self.device, self.compute_dtype, skip=("norm",))
        for p in todo:
            self._embeds[p] = self._encode(tokenizer, body, p)
        del tokenizer, text_encoder, body
        free_memory()
        self.log(f"  done in {time.time() - t:.0f} s")

    def _encode(self, tokenizer, body, prompt, max_length=512):
        """Same maths as Flux2KleinPipeline._get_qwen3_prompt_embeds."""
        import torch
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                           max_length=max_length)
        with torch.no_grad():
            out = body(input_ids=inputs["input_ids"].to(self.device),
                       attention_mask=inputs["attention_mask"].to(self.device),
                       output_hidden_states=True, use_cache=False)
        hidden = torch.stack([out.hidden_states[k] for k in TEXT_LAYERS], dim=1)
        b, c, s, d = hidden.shape
        return hidden.permute(0, 2, 1, 3).reshape(b, s, c * d).to(self.compute_dtype).cpu()

    # ---------- pictures ----------
    def _run(self, prompt, references, seed, width, height, steps):
        import torch
        self.prepare([prompt])
        self._ensure_image_model()
        t = time.time()

        def step_done(pipe, i, timestep, kwargs):
            self.log(f"  step {i + 1}/{steps} ({time.time() - t:.0f} s)")
            return kwargs

        kwargs = dict(prompt_embeds=self._embeds[prompt].to(self.device), width=width, height=height,
                      num_inference_steps=steps, guidance_scale=1.0,
                      generator=torch.Generator("cpu").manual_seed(seed),
                      callback_on_step_end=step_done)
        if references:
            kwargs["image"] = references
        image = self.pipe(**kwargs).images[0]
        self.log(f"  picture done in {time.time() - t:.0f} s")
        return image

    def create(self, prompt, seed=1, width=WIDTH, height=HEIGHT, steps=STEPS):
        """Text -> new character picture."""
        return self._run(prompt, None, seed, width, height, steps)

    def edit(self, prompt, references, seed=1, width=WIDTH, height=HEIGHT, steps=STEPS):
        """Reference picture(s) + instruction -> new picture of the same character."""
        refs = [prepare_reference(r) for r in references]
        return self._run(prompt, refs, seed, width, height, steps)


def prepare_reference(img, bg=(234, 244, 74), max_side=1024):
    """RGB copy of a reference picture; transparent areas are filled with the background colour."""
    if isinstance(img, str):
        img = Image.open(img)
    img.load()
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, bg)
        flat.paste(img, mask=img.getchannel("A"))
        img = flat
    else:
        img = img.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


_bg_session = None


def remove_background(img):
    """Transparent-background copy of a picture (local AI cut-out), or None if unavailable."""
    global _bg_session
    try:
        from rembg import new_session, remove
    except ImportError:
        return None
    try:
        if _bg_session is None:
            os.makedirs(os.environ["U2NET_HOME"], exist_ok=True)
            _bg_session = new_session("birefnet-general")
        return remove(img.convert("RGB"), session=_bg_session, post_process_mask=True)
    except Exception as exc:  # cut-out is a bonus; never lose the main picture over it
        print(f"  (background removal failed: {exc})")
        return None
