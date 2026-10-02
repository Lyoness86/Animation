"""Video export via FFmpeg (bundled through imageio-ffmpeg)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .imageio_utils import write_png
from .scene import to_bgr_over, to_straight_uint8

FORMATS = {
    # id: (label, extension, has_alpha)
    "mov": ("MOV with transparency (ProRes 4444) - recommended for CapCut", ".mov", True),
    "webm": ("WebM with transparency (VP9)", ".webm", True),
    "png": ("PNG image sequence (folder, with transparency)", "", True),
    "green": ("MP4 on green screen (use CapCut's Chroma key)", ".mp4", False),
    "mp4": ("MP4 (normal video, no transparency)", ".mp4", False),
}


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        raise RuntimeError("FFmpeg not found. Run setup.bat again.")


def _codec_args(fmt):
    if fmt == "mov":
        return ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
                "-vendor", "apl0"]
    if fmt == "webm":
        return ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "28",
                "-row-mt", "1", "-deadline", "good", "-cpu-used", "4", "-auto-alt-ref", "0"]
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium",
            "-movflags", "+faststart"]


def export_video(scene, out_path, fmt="mov", with_background=False, only=None,
                 width=1920, height=1080, fps=30, duration=None, progress=None,
                 cancelled=lambda: False):
    """Render frames and encode. `only` = render one character alone.
    progress(fraction) is called as frames are written. Returns output path."""
    duration = duration or scene.duration()
    n = max(1, int(round(duration * fps)))
    out_path = Path(out_path)
    alpha = FORMATS[fmt][2] and not with_background

    if fmt == "png":
        out_path.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            if cancelled():
                return None
            frame, _ = scene.render(i / fps, width, height, background=with_background, only=only)
            img = to_straight_uint8(frame) if alpha else to_bgr_over(frame)
            write_png(out_path / f"frame_{i:05d}.png", img)
            if progress:
                progress((i + 1) / n)
        return out_path

    pix_in = "bgra" if alpha else "bgr24"
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{width}x{height}", "-r", str(fps),
           "-i", "-", *_codec_args(fmt), str(out_path)]
    flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    try:
        for i in range(n):
            if cancelled():
                proc.stdin.close()
                proc.wait()
                try:
                    os.remove(out_path)
                except OSError:
                    pass
                return None
            frame, _ = scene.render(i / fps, width, height, background=with_background, only=only)
            if alpha:
                img = to_straight_uint8(frame)
            elif fmt == "green" and not with_background:
                img = to_bgr_over(frame, (0, 255, 0))
            else:
                img = to_bgr_over(frame)
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
            if progress:
                progress((i + 1) / n)
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError("FFmpeg failed:\n" + err[-2000:])
    except BrokenPipeError:
        err = proc.stderr.read().decode(errors="replace")
        raise RuntimeError("FFmpeg stopped unexpectedly:\n" + err[-2000:])
    return out_path
