from __future__ import annotations

import hashlib
import json
from pathlib import Path


def discover_blif_designs(dataset_root: Path) -> dict[str, Path]:
    dataset_root = dataset_root.resolve()
    designs: dict[str, Path] = {}
    duplicate_names: dict[str, list[Path]] = {}
    for blif_path in sorted(dataset_root.rglob("*.blif")):
        design_name = blif_path.name
        resolved = blif_path.resolve()
        if design_name in designs:
            duplicate_names.setdefault(design_name, [designs[design_name]]).append(resolved)
            continue
        designs[design_name] = resolved
    if duplicate_names:
        duplicate_summary = ", ".join(
            f"{name} ({'; '.join(str(path.relative_to(dataset_root)) for path in paths)})"
            for name, paths in sorted(duplicate_names.items())
        )
        raise ValueError(
            "Duplicate .blif filenames found under dataset root. "
            f"Use a narrower --dataset-root folder. Conflicts: {duplicate_summary}"
        )
    return designs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(dataset_root: Path) -> dict[str, object]:
    designs = discover_blif_designs(dataset_root)
    return {
        "dataset_root": str(dataset_root.resolve()),
        "designs": [
            {
                "name": name,
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in designs.items()
        ],
    }


def write_manifest(dataset_root: Path, output_path: Path) -> dict[str, object]:
    manifest = build_manifest(dataset_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    return manifest
