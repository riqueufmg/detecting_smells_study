#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


RE_DETECTION = re.compile(r"\b(detection|detected)\b\s*[:=]\s*(true|false|0|1)\b", re.IGNORECASE)


def read_json_from_txt(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Could not find JSON object inside file: {path}")
        return json.loads(raw[start : end + 1])


def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        v = x.strip().lower()
        if v in ("true", "1", "yes", "y"):
            return True
        if v in ("false", "0", "no", "n"):
            return False
    raise ValueError(f"Invalid boolean value for detection: {x!r}")


def pick_entity_from_obj(obj: Dict[str, Any]) -> Tuple[str, str]:
    if "package" in obj and str(obj["package"]).strip():
        return "package", str(obj["package"]).strip()
    if "class" in obj and str(obj["class"]).strip():
        return "class", str(obj["class"]).strip()
    return "unknown", ""


def entity_from_filename(file_path: Path) -> str:
    stem = file_path.stem
    if "_" not in stem:
        return stem
    return stem.split("_")[-1]


def try_extract_detection_from_text(text: str) -> Optional[int]:
    m = RE_DETECTION.search(text)
    if not m:
        return None
    v = m.group(2).strip().lower()
    if v in ("true", "1"):
        return 1
    if v in ("false", "0"):
        return 0
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smell", default="god_component")
    parser.add_argument("--llm", default="gpt")
    parser.add_argument("--base-dir", default="data/processed/llm_outputs")
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    in_dir = Path(args.base_dir) / args.smell / args.llm
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    out_path = Path(args.out) if args.out else (in_dir / "outputs.csv")

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for txt_path in sorted(in_dir.glob("*.txt")):
        raw_text = ""
        file_name = txt_path.name

        try:
            raw_text = txt_path.read_text(encoding="utf-8").strip()

            obj = read_json_from_txt(txt_path)

            smell_val = str(obj.get("smell", "")).strip()
            entity_type, entity_val = pick_entity_from_obj(obj)

            detection_bool = to_bool(obj.get("detection", False))
            detection_val = int(detection_bool)

            justification_val = str(obj.get("justification", "")).strip()

            rows.append(
                {
                    "file_name": file_name,
                    "smell": smell_val,
                    "entity_type": entity_type,
                    "entity": entity_val,
                    "detection": detection_val,
                    "justification": justification_val,
                }
            )

        except Exception as e:
            try:
                if not raw_text:
                    raw_text = txt_path.read_text(encoding="utf-8").strip()

                detection_fallback = try_extract_detection_from_text(raw_text)

                rows.append(
                    {
                        "file_name": file_name,
                        "smell": args.smell,
                        "entity_type": "class",
                        "entity": entity_from_filename(txt_path),
                        "detection": "" if detection_fallback is None else detection_fallback,
                        "justification": raw_text,
                    }
                )

                errors.append(f"{file_name}: {e} (kept via fallback)")

            except Exception as e2:
                errors.append(f"{file_name}: {e} (fallback failed: {e2})")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "smell",
                "entity_type",
                "entity",
                "detection",
                "justification",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Processed {len(rows)} rows.")
    print(f"[OK] CSV written to: {out_path}")

    if errors:
        print(f"[WARN] {len(errors)} issues:")
        for msg in errors[:30]:
            print("  -", msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())