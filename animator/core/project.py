"""Save / open animation projects.

A project is one portable ".puppet" file - a zip archive containing:
  project.json                 everything about the scene (see below)
  background.png / background.<ext>   the background image or video
  characters/<n>/<view>.png    each character view's cut-out picture
  characters/<n>/props/<k>.png held objects

Puppets are rebuilt from the cut-out picture + the checked joint dots when a
project is opened, so the file holds no program-specific data and doesn't
depend on any folder on the computer that saved it."""
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .character import Action, Character, Prop
from .rig import Rig
from .scene import Scene

FORMAT = "puppet-animator-project"
VERSION = 1
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"}


def _png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode image")
    return buf.tobytes()


def _png_read(z, name):
    img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Damaged image in project: {name}")
    return img


def _action_to_dict(a):
    d = {"kind": a.kind}
    if a.kind == "idle":
        d["duration"] = a.duration
    if a.target is not None:
        d["target"] = list(a.target)
    if a.view:
        d["view"] = a.view
    return d


def save_project(scene, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".saving")
    data = {"format": FORMAT, "version": VERSION,
            "scene": {"width": scene.width, "height": scene.height, "background": None},
            "characters": []}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        if scene.video is not None:
            ext = Path(scene.video.path).suffix.lower() or ".mp4"
            z.write(scene.video.path, "background" + ext, compress_type=zipfile.ZIP_STORED)
            data["scene"]["background"] = {"type": "video", "file": "background" + ext}
        elif scene.background is not None:
            z.writestr("background.png", _png_bytes(scene.background))
            data["scene"]["background"] = {"type": "image", "file": "background.png"}

        for n, ch in enumerate(scene.characters):  # list order = layer order (back first)
            views = {}
            for view, rig in ch.views.items():
                name = f"characters/{n}/{view}.png"
                z.writestr(name, _png_bytes(rig.image))
                views[view] = {"image": name, "joints": {k: [float(v[0]), float(v[1])] for k, v in rig.joints.items()},
                               "faces_left": bool(rig.faces_left), "name": rig.name}
            props = []
            for k, pr in enumerate(ch.props):
                name = f"characters/{n}/props/{k}.png"
                z.writestr(name, _png_bytes(pr.image))
                props.append({"image": name, "name": pr.name, "hand": pr.hand, "size": pr.size,
                              "offset": list(pr.offset), "rotation": pr.rotation, "layer": pr.layer,
                              "grip": list(pr.grip), "follow": pr.follow})
            data["characters"].append({
                "name": ch.name, "views": views, "start_view": ch.start_view,
                "position": list(ch.pos), "height": ch.height, "mirrored": ch.mirrored,
                "actions": [_action_to_dict(a) for a in ch.actions], "props": props,
                "side_view_walk": ch.side_view_walk, "alive": ch.alive})
        z.writestr("project.json", json.dumps(data, indent=2))
    tmp.replace(path)
    return path


def load_project(path, progress=None):
    """Returns a Scene. progress(text) is called while puppets are rebuilt."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("project.json"))
        if data.get("format") != FORMAT:
            raise ValueError("This is not a Puppet Animator project.")
        if data.get("version", 0) > VERSION:
            raise ValueError("This project was saved by a newer version of the program.")
        sd = data["scene"]
        scene = Scene(sd.get("width", 1920), sd.get("height", 1080))
        bg = sd.get("background")
        if bg and bg["type"] == "image":
            scene.set_background(_png_read(z, bg["file"]))
        elif bg and bg["type"] == "video":
            folder = Path(tempfile.mkdtemp(prefix="puppet_bg_"))
            target = folder / Path(bg["file"]).name
            with z.open(bg["file"]) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            scene.set_background_video(target)

        for cd in data["characters"]:
            rigs = {}
            for view, vd in cd["views"].items():
                if progress:
                    progress(f"Rebuilding {cd['name']} ({view})...")
                rigs[view] = Rig(_png_read(z, vd["image"]), {k: tuple(v) for k, v in vd["joints"].items()},
                                 vd.get("name", cd["name"]), faces_left=vd.get("faces_left", False),
                                 profile=(view == "side"))
            props = [Prop(_png_read(z, p["image"]), p.get("name", "Object"), p.get("hand", "r"),
                          p.get("size", 0.15), tuple(p.get("offset", (0, 0))), p.get("rotation", 0.0),
                          p.get("layer", 2), tuple(p.get("grip", (0.5, 0.6))), p.get("follow", 0.0))
                     for p in cd.get("props", [])]
            actions = [Action(a["kind"], a.get("duration", 0.0),
                              tuple(a["target"]) if a.get("target") is not None else None, a.get("view"))
                       for a in cd.get("actions", [])]
            ch = Character(rigs.get("front") or next(iter(rigs.values())), tuple(cd["position"]),
                           cd["height"], cd.get("mirrored", False), actions, rigs,
                           cd.get("start_view", "front"), props, cd.get("name", ""),
                           cd.get("side_view_walk", True), cd.get("alive", 1.0))
            scene.characters.append(ch)
    return scene
