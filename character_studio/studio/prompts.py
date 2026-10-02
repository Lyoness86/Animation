"""Prompt wording. Kept in one place so the style stays identical across images."""

STYLE = ("3D animated feature film style character, Pixar-like render, soft studio lighting, "
         "high detail, clean shapes")

FRAMING = ("full body visible from head to feet, centered, the whole character inside the "
           "picture with a margin around it, nothing cut off")

BACKGROUND = ("on a plain flat solid {colour} background, no floor, no shadow on the ground, "
              "no scenery, no other objects, no text")

# Name -> how to describe it to the model. The full app will let you add your own.
POSES = {
    "front": "standing relaxed, facing the viewer, arms by the sides",
    "three_quarter": "standing relaxed, body turned three-quarters to the left",
    "side": "standing relaxed, seen exactly from the side (profile view), facing left",
    "three_quarter_back": "standing relaxed, seen three-quarters from behind",
    "back": "standing relaxed, seen from directly behind, facing away from the viewer",
    "waving": "facing the viewer, smiling and waving with the right hand raised above the shoulder",
    "walking": "walking towards the left, seen from the side, mid-stride",
    "jumping": "jumping happily in the air, both feet off the ground, arms up",
    "sitting": "sitting on an invisible seat, facing the viewer, hands on knees",
    "pointing": "facing the viewer, pointing to the left with the right arm stretched out",
    "drinking": "facing the viewer, drinking from a coffee mug held in the right hand",
}


def create_prompt(description, bg_colour):
    return (f"{description}. {STYLE}. Standing relaxed, facing the viewer, arms slightly away "
            f"from the body. {FRAMING}, {BACKGROUND.format(colour=bg_colour)}.")


def pose_prompt(pose, bg_colour, description=""):
    action = POSES.get(pose, pose)
    who = f" ({description})" if description else ""
    return (f"The exact same character as in the reference image{who}, now {action}. "
            "Keep the identical face, eyes, hair style and hair colour, skin, clothing, "
            "shoes, colours, body shape and proportions. Same 3D animated style. "
            f"{FRAMING}, {BACKGROUND.format(colour=bg_colour)}.")
