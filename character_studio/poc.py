"""Proof of concept: make one character, then the same character in other poses.

  python poc.py --describe "a cheerful pig girl with long bright blue hair, mint t-shirt, jeans"
  python poc.py --reference "C:\\pictures\\Leah no drink.png"

Everything runs locally on this PC. Pictures go to output\\<name>_<time>\\.
"""
import argparse
import datetime
import json
import os
import re
import sys
import time

from PIL import Image

from studio import prompts
from studio.engine import Engine, free_memory, prepare_reference, remove_background

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POSES = ["waving", "walking", "back"]
BG_COLOUR = "bright yellow"   # same kind of plain background the animation programme already keys


def contact_sheet(images, labels, height=512):
    from PIL import ImageDraw
    thumbs = [im.convert("RGB").resize((round(im.width * height / im.height), height))
              for im in images]
    sheet = Image.new("RGB", (sum(t.width for t in thumbs) + 10 * (len(thumbs) + 1), height + 50),
                      "white")
    draw = ImageDraw.Draw(sheet)
    x = 10
    for thumb, label in zip(thumbs, labels):
        sheet.paste(thumb, (x, 40))
        draw.text((x + 4, 12), label, fill="black")
        x += thumb.width + 10
    return sheet


def save_cutouts(folder, stems):
    """Transparent copies, made after the image model is unloaded (they need RAM too)."""
    print("Cutting out the backgrounds...")
    for stem in stems:
        cut = remove_background(Image.open(os.path.join(folder, f"{stem}.png")))
        if cut is not None:
            cut.save(os.path.join(folder, f"{stem}_transparent.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--describe", help="text description of a new character")
    src.add_argument("--reference", help="existing character picture to use as the reference")
    ap.add_argument("--poses", nargs="+", default=DEFAULT_POSES,
                    help=f"poses to try (known: {', '.join(prompts.POSES)}), or your own words")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--name", default=None)
    ap.add_argument("--small", action="store_true",
                    help="smaller pictures (576x768): about twice as fast, for quick tests")
    ap.add_argument("--no-open", action="store_true", help="don't open the folder at the end")
    args = ap.parse_args(argv)

    name = args.name or (os.path.splitext(os.path.basename(args.reference))[0] if args.reference
                         else " ".join(args.describe.split()[:4]))
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:40] or "character"
    folder = os.path.join(HERE, "output", f"{slug}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(folder, exist_ok=True)

    engine = Engine()
    t0 = time.time()
    description = args.describe or ""
    size = dict(width=576, height=768) if args.small else {}
    create_prompt = prompts.create_prompt(description, BG_COLOUR)
    pose_prompts = [prompts.pose_prompt(p, BG_COLOUR, description) for p in args.poses]
    # all descriptions are read in one go, so small PCs load the text model only once
    engine.prepare(([] if args.reference else [create_prompt]) + pose_prompts)
    stems = []

    if args.reference:
        print(f"Using your picture as the character: {args.reference}")
        base = prepare_reference(args.reference)
        base.save(os.path.join(folder, "00_reference.png"))
    else:
        print("Step 1: creating the character from your description...")
        base = engine.create(create_prompt, seed=args.seed, **size)
        base.save(os.path.join(folder, "00_character.png"))
        stems.append("00_character")

    images, labels = [base], ["character" if args.describe else "reference"]
    for i, (pose, prompt) in enumerate(zip(args.poses, pose_prompts), 1):
        print(f"Step {i + 1}: same character, pose '{pose}'...")
        img = engine.edit(prompt, [base], seed=args.seed, **size)
        stem = f"{i:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', pose)[:30]}"
        img.save(os.path.join(folder, f"{stem}.png"))
        stems.append(stem)
        images.append(img)
        labels.append(pose)
    del engine
    free_memory()

    contact_sheet(images, labels).save(os.path.join(folder, "comparison_sheet.png"))
    save_cutouts(folder, stems)
    with open(os.path.join(folder, "character.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "description": description, "reference": args.reference,
                   "seed": args.seed, "background": BG_COLOUR, "poses": args.poses,
                   "model": "FLUX.2 [klein] 4B"}, f, indent=2)

    print(f"\nFinished in {time.time() - t0:.0f} s. Pictures saved in:\n  {folder}")
    print("Open comparison_sheet.png to judge whether it is the same character.")
    if os.name == "nt" and not args.no_open:
        os.startfile(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
