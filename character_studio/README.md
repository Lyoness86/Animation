# Character Studio — proof of concept

A separate programme from the Puppet Animator. It **creates** character pictures
on your own PC with a local, open-source AI model. Nothing is sent to an online
service. It only goes online to download the model once.

This is only the **proof of concept**: it makes one character and then tries to
draw the *same* character in other poses, so we can judge the consistency
before building the full application.

## What it uses

| Part | What | Size |
|---|---|---|
| Image model | **FLUX.2 [klein] 4B** (Black Forest Labs, Apache 2.0, free for commercial use). One model creates characters from text **and** redraws a reference picture in a new pose. | ~16 GB |
| Background cut-out | **BiRefNet** via `rembg` (local) | ~1 GB |
| Python + PyTorch (CUDA) | in a private `.venv` folder | ~6 GB |

**Total disk: about 25 GB (allow 30 GB).** Everything stays inside this
folder (`.venv`, `models`, `output`). To uninstall, delete the folder.

## Hardware

| Graphics card (NVIDIA) | Result |
|---|---|
| 16 GB+ VRAM (RTX 4070 Ti Super / 4080 / 4090 / 5070 Ti / 5080 / 5090, 3090) | Excellent, ~5–20 s per picture |
| 12 GB (RTX 3060 12 GB, 4070, 4070 Super, 5070) | Good, ~15–45 s per picture |
| 8 GB (RTX 3060 Ti, 3070, 4060, 4060 Ti 8 GB) | Works but slow, ~1–4 min per picture |
| under 8 GB, AMD / Intel graphics, or no graphics card | Not practical |

RAM: 16 GB minimum, **32 GB recommended**.

## Try it

1. Double-click **`check_my_pc.bat`** and send the text it shows to Claude.
2. If the verdict is not "NOT PRACTICAL", double-click **`setup.bat`** (once;
   downloads ~20 GB).
3. Either:
   - **drag one of your existing character pictures** (e.g. `Leah no drink.png`)
     onto **`run_poc.bat`**, or
   - double-click **`run_poc.bat`** and type a description.
4. When it finishes, it opens a folder in `output\`. Look at
   **`comparison_sheet.png`**: the first picture is the character, and the others
   are the same character waving, walking and seen from behind. Each picture is
   saved twice: with a plain yellow background (`.png`) and with a transparent
   background (`_transparent.png`). Both work in the animation programme.

Advanced (from a command prompt in this folder):

```
.venv\Scripts\python poc.py --describe "..." --poses waving jumping "sitting on a bench" --seed 3
```

## For developers

- `studio/engine.py`: loads the model, chooses a memory mode from the VRAM,
  `create()` (text to picture), `edit()` (references + text to picture) and the background cut-out.
- `studio/prompts.py`: style, framing and pose wording.
- `poc.py`: the proof-of-concept flow.
- `check_pc.py`: hardware check (standard library only).
- Tests: `python -m pytest tests`. These run the whole flow with a tiny random-weight
  model on the CPU, so they test the plumbing but not picture quality.
