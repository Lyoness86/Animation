"""Local image generation with FLUX.2 [klein] 4B (runs entirely on this PC).

One model does both jobs:
  * text -> new character
  * reference picture(s) + text -> the same character in a new pose/angle
"""
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


def gpu_info():
    """(name, vram_gb) of the first CUDA GPU, or (None, 0)."""
    import torch
    if not torch.cuda.is_available():
        return None, 0.0
    props = torch.cuda.get_device_properties(0)
    return props.name, props.total_memory / 1024 ** 3


def pick_memory_mode(vram_gb):
    """How much of the model to keep on the graphics card at once.

    The whole model is ~16 GB, so only big cards hold all of it. Smaller cards
    move parts in and out of graphics memory (slower, but works).
    """
    if vram_gb >= 22:
        return "gpu"
    if vram_gb >= 11.5:
        return "model_offload"
    return "sequential_offload"


class Engine:
    def __init__(self, pipe=None, device=None, log=print):
        self.log = log
        self.pipe = pipe
        self.device = device
        if pipe is None:
            self._load()

    def _load(self):
        import torch
        from diffusers import Flux2KleinPipeline

        name, vram = gpu_info()
        if name is None:
            raise RuntimeError(
                "No NVIDIA graphics card is available to Python. Run check_my_pc.bat; "
                "if it shows an NVIDIA card, update its driver and run setup.bat again.")
        mode = pick_memory_mode(vram)
        self.log(f"Graphics card: {name} ({vram:.0f} GB) - memory mode: {mode}")
        self.log("Loading the model (the first time this downloads about 16 GB)...")
        t = time.time()
        pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        if mode == "gpu":
            pipe.to("cuda")
        elif mode == "model_offload":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.device = "cuda"
        self.log(f"Model ready in {time.time() - t:.0f} s")

    def _run(self, prompt, references, seed, width, height, steps):
        import torch
        t = time.time()
        kwargs = dict(prompt=prompt, width=width, height=height, num_inference_steps=steps,
                      guidance_scale=1.0,
                      generator=torch.Generator("cpu").manual_seed(seed))
        if references:
            kwargs["image"] = references
        image = self.pipe(**kwargs).images[0]
        self.log(f"  done in {time.time() - t:.0f} s")
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
