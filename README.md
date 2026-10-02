# Puppet Animator (prototype)

Turn a character picture on a plain-colour background into a puppet you can
animate, place on a background, and export as transparent video for CapCut.

## Install (Windows, once)

1. Install **Python 3.12** from https://www.python.org/downloads/ - on the first
   installer screen tick **"Add python.exe to PATH"**.
2. Download this project (green **Code** button on GitHub -> *Download ZIP*) and unzip it.
3. Double-click **`setup.bat`** and wait (a few minutes; it downloads ~300 MB).

## Use

Double-click **`run.bat`**.

1. **Background image...** - pick a background picture.
2. **Add character...** - pick a character picture with a plain background
   (blue, green, yellow... any single colour that isn't in the character).
   The program removes the background, finds the body automatically and shows
   the coloured body parts. If a dot is clearly wrong, drag it; otherwise click
   **Looks good**.
3. Drag the character to place it; mouse wheel over it to resize.
   **Bring forward / Send back** changes which character is in front.
4. With a character selected, add steps: **Walk to...** (then click the spot
   where it should stop), **Stand still**, **Wave**, **Jump**, **Stop**.
   Each character has its own list; they all start at the same time.
5. **Play** to preview.
6. **Export video...** - either the whole scene, or each character as its own
   transparent video (MOV ProRes 4444 is the best choice for CapCut; if CapCut
   doesn't show transparency use "MP4 on green screen" + CapCut's *Chroma key*).

## Projects

**Save project** writes one `.puppet` file containing everything: background
(image or video), characters (pictures + checked dots), positions, sizes,
layer order, flips, views, held objects and every character's step list.
Copy it to any PC and **Open project...** it there. `examples/garden_demo.puppet`
is a ready-made example.

## Views (turning)

Select a character, **Add view (side, back...)**, pick which view the picture
shows (3/4, side, 3/4 back, back), then add a **Turn to view...** step. The
character spins (quick squeeze) and swaps picture halfway. Walking in a side
view looks much better than in the front view. "Starts:" chooses the first view.

## Held objects

**Hold test glass** or **Hold object...** (any picture on a plain background).
Choose which hand (by side of the picture), size, position, rotation and
whether it is drawn behind the body, behind the hand, in front of the hand or
in front of everything. The **Drink** animations lift it to the mouth.

## Adding animations

Animations are plain text files in `animator/animations/`. Copy one, change
the numbers, save it with a new name - it appears as a new button. See the
top of `animator/core/clips.py` for the format.

## For developers

- `animator/core/` - engine (no UI): `keying.py` background removal,
  `skeleton.py` joint detection, `rig.py` automatic body-part split,
  `clips.py` animation files, `character.py` per-character action lists,
  `scene.py` rendering, `export.py` FFmpeg export.
- `animator/ui/` - PySide6 interface.
- Tests: `python -m pytest tests`; `python tests/render_sheet.py` renders
  contact sheets; `python tests/debug_rig.py IMAGE` shows the body-part split
  for an image in `test_output/`.
