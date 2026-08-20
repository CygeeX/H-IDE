#!/usr/bin/env python3
"""Show the live MuJoCo head-camera JPEG in a local read-only window."""

from __future__ import annotations

from io import BytesIO
import json
import tkinter as tk
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from PIL import Image, ImageTk


PLANT_BASE_URL = "http://127.0.0.1:8794"
REFRESH_MS = 200


class FirstPersonViewer:
    def __init__(self) -> None:
        self._opener = build_opener(ProxyHandler({}))
        self._root = tk.Tk()
        self._root.title("H-IDE MuJoCo · First-person RGB")
        self._root.geometry("680x560+2120+120")
        self._root.configure(background="#17151c")
        self._image_label = tk.Label(
            self._root,
            background="#17151c",
            borderwidth=0,
        )
        self._image_label.pack(padx=20, pady=(20, 8))
        self._status = tk.StringVar(value="Connecting to MuJoCo RGB-D…")
        tk.Label(
            self._root,
            textvariable=self._status,
            anchor="w",
            justify="left",
            foreground="#f1e9dc",
            background="#17151c",
            font=("Sans", 11),
        ).pack(fill="x", padx=20, pady=(0, 16))
        self._photo: ImageTk.PhotoImage | None = None

    def _read(self, path: str) -> bytes:
        with self._opener.open(f"{PLANT_BASE_URL}{path}", timeout=0.8) as response:
            if response.status != 200:
                raise URLError(f"unexpected HTTP status {response.status}")
            return response.read()

    def _refresh(self) -> None:
        try:
            image = Image.open(BytesIO(self._read("/rgb.jpg"))).convert("RGB")
            image.thumbnail((640, 480), Image.Resampling.LANCZOS)
            self._photo = ImageTk.PhotoImage(image)
            self._image_label.configure(image=self._photo)
            state = json.loads(self._read("/state"))
            self._status.set(
                "LIVE · head_camera · "
                f"sequence {state['sequence']} · "
                f"sim {float(state['simulationTimeS']):.2f}s · "
                f"capture {float(state['captureAgeMs']):.1f}ms"
            )
        except (KeyError, OSError, TypeError, ValueError, URLError, json.JSONDecodeError) as exc:
            self._status.set(f"MuJoCo first-person stream unavailable: {exc}")
        finally:
            self._root.after(REFRESH_MS, self._refresh)

    def run(self) -> None:
        self._root.after(0, self._refresh)
        self._root.mainloop()


def main() -> int:
    FirstPersonViewer().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
