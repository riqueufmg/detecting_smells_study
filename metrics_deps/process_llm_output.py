#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

RE_DETECTION = re.compile(r"\b(detection|detected)\b\s*[:=]\s*(true|false|0|1)\b", re.IGNORECASE)
PACKAGE_LEVEL = {"god_component", "unstable_dependency"}


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
    parser = argparse.ArgumentParser(
        description="Consolidate all projects' LLM outputs into ONE CSV at llm_outputs level."
    )
    parser.add_argument("--smell", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--base-dir", default="data/processed/llm_outputs")
    parser.add_argument(
        "--out",
        default=None,
        help='Default: "{base-dir}/{smell}_{llm}.csv"',
    )
    parser.add_argument(
        "--include-path",
        action="store_true",
        help="If set, adds a file_path column with the full relative path.",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    # Find all matching folders: base_dir/<project>/<smell>/<llm>/
    target_folders = []
    for proj_dir in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        folder = proj_dir / args.smell / args.llm
        if folder.exists() and folder.is_dir():
            target_folders.append((proj_dir.name, folder))

    if not target_folders:
        print("[WARN] No matching folders found. Expected:")
        print("       base-dir/<project>/<smell>/<llm>/*.txt")
        print(f"       base-dir={base_dir}, smell={args.smell}, llm={args.llm}")
        return 1

    out_path = Path(args.out) if args.out else (base_dir / f"{args.smell}_{args.llm}.csv")

    fallback_entity_type = "package" if args.smell in PACKAGE_LEVEL else "class"

    rows: List[Dict[str, Any]] = []
    issues: List[str] = []

    for project_name, folder in target_folders:
        for txt_path in sorted(folder.glob("*.txt")):
            file_name = txt_path.name
            raw_text = ""

            try:
                raw_text = txt_path.read_text(encoding="utf-8").strip()
                obj = read_json_from_txt(txt_path)

                smell_val = str(obj.get("smell", "")).strip()
                entity_type, entity_val = pick_entity_from_obj(obj)
                detection_val = int(to_bool(obj.get("detection", False)))
                justification_val = str(obj.get("justification", "")).strip()

                row = {
                    "project": project_name,
                    "file_name": file_name,
                    "smell": smell_val,
                    "entity_type": entity_type,
                    "entity": entity_val,
                    "detection": detection_val,
                    "justification": justification_val,
                }
                if args.include_path:
                    row["file_path"] = str(txt_path.relative_to(base_dir))

                rows.append(row)

            except Exception as e:
                # fallback: keep file anyway
                try:
                    if not raw_text:
                        raw_text = txt_path.read_text(encoding="utf-8").strip()

                    detection_fb = try_extract_detection_from_text(raw_text)
                    row = {
                        "project": project_name,
                        "file_name": file_name,
                        "smell": args.smell,
                        "entity_type": fallback_entity_type,
                        "entity": entity_from_filename(txt_path),
                        "detection": "" if detection_fb is None else detection_fb,
                        "justification": raw_text,
                    }
                    if args.include_path:
                        row["file_path"] = str(txt_path.relative_to(base_dir))

                    rows.append(row)
                    issues.append(f"{project_name}/{file_name}: {e} (kept via fallback)")

                except Exception as e2:
                    issues.append(f"{project_name}/{file_name}: {e} (fallback failed: {e2})")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["project", "file_name", "smell", "entity_type", "entity", "detection", "justification"]
    if args.include_path:
        fieldnames.insert(2, "file_path")  # after file_name (optional)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Projects matched: {len(target_folders)}")
    print(f"[OK] Rows written: {len(rows)}")
    print(f"[OK] CSV written to: {out_path}")
    if issues:
        print(f"[WARN] Issues: {len(issues)} (non-JSON kept via fallback when possible)")
        for msg in issues[:30]:
            print("  -", msg)
        if len(issues) > 30:
            print("  ... (more omitted)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())