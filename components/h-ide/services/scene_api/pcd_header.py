from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MAX_HEADER_BYTES = 64 * 1024
ENCODINGS = {"ascii", "binary", "binary_compressed"}


class PcdHeaderError(ValueError):
    pass


@dataclass(frozen=True)
class PcdHeader:
    fields: tuple[str, ...]
    sizes: tuple[int, ...]
    counts: tuple[int, ...]
    points: int
    data_encoding: str
    decompressed_bytes: int


def parse_pcd_header(path: Path, max_header_bytes: int = MAX_HEADER_BYTES) -> PcdHeader:
    with path.open("rb") as stream:
        prefix = stream.read(max_header_bytes + 1)

    data_line_indexes: list[int] = []
    lines = prefix.splitlines(keepends=True)
    consumed = 0
    values: dict[str, list[str]] = {}
    for index, raw_line in enumerate(lines):
        consumed += len(raw_line)
        if consumed > max_header_bytes:
            break
        try:
            line = raw_line.decode("ascii").strip()
        except UnicodeDecodeError:
            line = ""
        if not line or line.startswith("#"):
            continue
        pieces = line.split()
        key = pieces[0].upper()
        if key == "DATA":
            data_line_indexes.append(index)
            if len(pieces) != 2:
                raise PcdHeaderError("malformed DATA directive")
            values[key] = pieces[1:]
            # A second DATA immediately after the first is never point data.
            for following in lines[index + 1 :]:
                try:
                    following_text = following.decode("ascii").strip()
                except UnicodeDecodeError:
                    break
                if following_text.upper().startswith("DATA "):
                    raise PcdHeaderError("duplicate DATA directive")
                if following_text and not following_text.startswith("#"):
                    break
            break
        if key in values and key in {"FIELDS", "SIZE", "COUNT", "POINTS"}:
            raise PcdHeaderError(f"duplicate {key} directive")
        values[key] = pieces[1:]

    if not data_line_indexes:
        raise PcdHeaderError("DATA directive missing or header exceeds 64 KiB")
    encoding = values["DATA"][0].lower()
    if encoding not in ENCODINGS:
        raise PcdHeaderError("unsupported DATA encoding")

    fields = tuple(item.lower() for item in values.get("FIELDS", []))
    if not {"x", "y", "z"}.issubset(fields):
        raise PcdHeaderError("FIELDS must contain x, y and z")
    try:
        sizes = tuple(int(item) for item in values["SIZE"])
        count_values = values.get("COUNT")
        counts = tuple(int(item) for item in count_values) if count_values else (1,) * len(fields)
        points_values = values["POINTS"]
        if len(points_values) != 1:
            raise ValueError
        points = int(points_values[0])
    except (KeyError, TypeError, ValueError) as error:
        raise PcdHeaderError("malformed SIZE, COUNT or POINTS") from error
    if len(sizes) != len(fields) or len(counts) != len(fields):
        raise PcdHeaderError("FIELDS, SIZE and COUNT lengths differ")
    if points <= 0 or any(value <= 0 for value in (*sizes, *counts)):
        raise PcdHeaderError("SIZE, COUNT and POINTS must be positive")
    decompressed = points * sum(size * count for size, count in zip(sizes, counts, strict=True))
    return PcdHeader(fields, sizes, counts, points, encoding, decompressed)
