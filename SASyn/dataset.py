from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _relative_design_name(dataset_root: Path, blif_path: Path) -> str:
    return blif_path.relative_to(dataset_root).as_posix()


def discover_blif_designs(dataset_root: Path) -> dict[str, Path]:
    dataset_root = dataset_root.resolve()
    designs: dict[str, Path] = {}
    candidates = sorted(dataset_root.rglob("*.blif"))
    name_counts: dict[str, int] = {}
    for blif_path in candidates:
        name_counts[blif_path.name] = name_counts.get(blif_path.name, 0) + 1
    for blif_path in candidates:
        design_name = blif_path.name
        if name_counts[design_name] > 1:
            design_name = _relative_design_name(dataset_root, blif_path)
        designs[design_name] = blif_path.resolve()
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
