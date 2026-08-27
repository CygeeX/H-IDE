#!/usr/bin/env python3
"""Show the live MuJoCo head-camera JPEG in a local read-only window."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import stat
import tkinter as tk
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

from PIL import Image, ImageTk


PLANT_BASE_URL = "http://127.0.0.1:8794"
# Plant publishes the authoritative RGB-D cache at 10 Hz.  Match that source
# rate so a strict 10 Hz terminal-settle trace can prove the first-person
# window remained live without polling duplicate frames faster than Plant can
# produce them.
REFRESH_MS = 100
STATUS_SCHEMA_VERSION = "h-ide.first-person-viewer-health/1"


def default_status_path() -> Path:
    runtime_root = Path(
        os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    )
    return (
        runtime_root
        / f"h-ide-mujoco-demo-{os.getuid()}"
        / "first-person-viewer-status.json"
    )


class ViewerStatusFile:
    """Atomically publish evidence produced by the Tk rendering process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.boot_id = str(uuid4())
        self.pid = os.getpid()
        self.sequence = 0
        self.failure_count = 0
        self._plant_boot_id: str | None = None
        self._plant_session_id: str | None = None
        self._last_plant_sequence: int | None = None
        self._last_simulation_time_s: float | None = None
        self._last_capture_age_ms: float | None = None
        self._last_success_at_unix_ns: int | None = None
        self._last_error: str | None = None
        self._status = "starting"
        self._mapped = False
        self._prepare_parent()
        self._publish()

    def _prepare_parent(self) -> None:
        parent = self.path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = parent.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            raise RuntimeError(
                "first-person viewer status directory ownership or permissions are unsafe"
            )
        try:
            target = self.path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(target.st_mode) or target.st_uid != os.geteuid():
            raise RuntimeError("first-person viewer status path is unsafe")

    @staticmethod
    def _finite(value: object, label: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{label} must be finite")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{label} must be finite")
        return result

    @staticmethod
    def _plant_sequence(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Plant capture sequence must be a non-negative integer")
        return value

    def record_success(self, state: object, *, mapped: bool) -> None:
        if not isinstance(state, dict):
            raise ValueError("Plant state must be a JSON object")
        plant_sequence = self._plant_sequence(state.get("sequence"))
        plant_boot_id = state.get("producerBootId")
        if not isinstance(plant_boot_id, str) or not plant_boot_id.strip():
            raise ValueError("Plant producer boot ID must be non-empty")
        plant_session_id = state.get("sessionId")
        if not isinstance(plant_session_id, str) or not plant_session_id.strip():
            raise ValueError("Plant session ID must be non-empty")
        if self._plant_boot_id is not None and plant_boot_id != self._plant_boot_id:
            raise ValueError("Plant producer boot ID changed")
        if (
            self._plant_session_id is not None
            and plant_session_id != self._plant_session_id
        ):
            raise ValueError("Plant session ID changed")
        simulation_time_s = self._finite(
            state.get("simulationTimeS"), "Plant simulation time"
        )
        capture_age_ms = self._finite(
            state.get("captureAgeMs"), "Plant capture age"
        )
        if capture_age_ms < 0.0:
            raise ValueError("Plant capture age cannot be negative")
        if (
            self._last_plant_sequence is not None
            and plant_sequence <= self._last_plant_sequence
        ):
            raise ValueError("Plant capture sequence did not advance")
        if not mapped:
            raise ValueError("first-person viewer window is not mapped")
        self.sequence += 1
        self._plant_boot_id = plant_boot_id
        self._plant_session_id = plant_session_id
        self._last_plant_sequence = plant_sequence
        self._last_simulation_time_s = simulation_time_s
        self._last_capture_age_ms = capture_age_ms
        self._last_success_at_unix_ns = time.time_ns()
        self._last_error = None
        self._status = "running"
        self._mapped = True
        self._publish()

    def record_failure(self, error: object, *, mapped: bool) -> None:
        self.failure_count += 1
        message = str(error).strip().replace("\x00", "")
        self._last_error = (message or type(error).__name__)[:512]
        self._status = "degraded"
        self._mapped = bool(mapped)
        self._publish()

    def close(self) -> None:
        self._status = "stopped"
        self._mapped = False
        self._publish()

    def _payload(self) -> dict[str, object]:
        return {
            "schemaVersion": STATUS_SCHEMA_VERSION,
            "bootId": self.boot_id,
            "pid": self.pid,
            "status": self._status,
            "healthy": self._status == "running" and self._mapped,
            "mapped": self._mapped,
            "sequence": self.sequence,
            "plantBootId": self._plant_boot_id,
            "plantSessionId": self._plant_session_id,
            "plantSequence": self._last_plant_sequence,
            "simulationTimeS": self._last_simulation_time_s,
            "captureAgeMs": self._last_capture_age_ms,
            "lastSuccessAtUnixNs": self._last_success_at_unix_ns,
            "failureCount": self.failure_count,
            "lastError": self._last_error,
        }

    def _publish(self) -> None:
        payload = json.dumps(
            self._payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        temporary = self.path.with_name(
            f".{self.path.name}.{self.pid}.{self.boot_id}.tmp"
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary, flags, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, self.path)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise


class FirstPersonViewer:
    def __init__(self, *, status_path: str | Path | None = None) -> None:
        self._opener = build_opener(ProxyHandler({}))
        self._health = ViewerStatusFile(status_path or default_status_path())
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
        self._closed = False
        self._root.protocol("WM_DELETE_WINDOW", self._close)

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
            mapped = bool(self._root.winfo_ismapped())
            self._health.record_success(state, mapped=mapped)
            self._status.set(
                "LIVE · head_camera · "
                f"sequence {state['sequence']} · "
                f"sim {float(state['simulationTimeS']):.2f}s · "
                f"capture {float(state['captureAgeMs']):.1f}ms"
            )
        except (KeyError, OSError, TypeError, ValueError, URLError, json.JSONDecodeError) as exc:
            self._status.set(f"MuJoCo first-person stream unavailable: {exc}")
            self._health.record_failure(
                exc,
                mapped=bool(self._root.winfo_ismapped()),
            )
        finally:
            if not self._closed:
                self._root.after(REFRESH_MS, self._refresh)

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._health.close()
        self._root.destroy()

    def run(self) -> None:
        try:
            self._root.after(0, self._refresh)
            self._root.mainloop()
        finally:
            self._close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", type=Path, default=default_status_path())
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    FirstPersonViewer(status_path=args.status_file).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
