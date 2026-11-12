"""Verify and identify one checkpoint, its sidecars and exact training source."""

from __future__ import annotations
import argparse
import hashlib
import json
import struct
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_package(checkpoint, source_snapshot):
    checkpoint, source_snapshot = Path(checkpoint).resolve(), Path(source_snapshot).resolve()
    before = checkpoint.stat()
    with checkpoint.open("rb") as stream:
        n = struct.unpack("<Q", stream.read(8))[0]
        if n > 16 * 1024 * 1024:
            raise ValueError("Oversized safetensors header")
        header = json.loads(stream.read(n))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    if not tensors or max(v["data_offsets"][1] for v in tensors) + 8 + n != before.st_size:
        raise ValueError("Safetensors extent mismatch")
    contract = json.loads((checkpoint.parent / "action_contract.json").read_text())
    modes = {
        "openwam-jenga-eef-delta-chunkstart-spatial-rot6d-v1": "eef_delta",
        "openwam-jenga-absolute-joint-fixed-j8-v1": "joint_absolute",
    }
    mode = modes[contract["version"]]
    if contract["horizon"] != 32 or contract["action_dim"] != 80:
        raise ValueError("Unsupported checkpoint dimensions")
    sidecars = {
        name: sha256(checkpoint.parent / name)
        for name in ("action_contract.json", "config.yaml", "normalization_stats.npy")
    }
    if sidecars["normalization_stats.npy"] != contract["norm_sha256"]:
        raise ValueError("Normalization contract mismatch")
    if mode == "eef_delta":
        sidecars["eef_delta_contract.py"] = sha256(checkpoint.parent / "eef_delta_contract.py")
    tokenizer = {
        str(p.relative_to(checkpoint.parent)): sha256(p)
        for p in sorted((checkpoint.parent / "tokenizer").rglob("*"))
        if p.is_file()
    }
    if not tokenizer:
        raise ValueError("Missing checkpoint tokenizer")
    source_manifest_path = source_snapshot.parent / "SOURCE_SHA256.json"
    source_expected = json.loads(source_manifest_path.read_text())
    source_hashes = {name: sha256(source_snapshot / name) for name in source_expected}
    if source_hashes != source_expected:
        bad = [name for name in source_expected if source_hashes[name] != source_expected[name]]
        raise ValueError(f"Training source freeze mismatch: {bad}")
    weights_sha = sha256(checkpoint)
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint changed during hashing")
    payload = dict(
        schema="tower-openwam-checkpoint-v1",
        mode=mode,
        checkpoint=str(checkpoint),
        checkpoint_sha256=weights_sha,
        checkpoint_bytes=before.st_size,
        tensor_count=len(tensors),
        sidecars_sha256=sidecars,
        tokenizer_sha256=tokenizer,
        source_snapshot=str(source_snapshot),
        source_sha256=source_hashes,
        action_contract=contract,
        inference_qualified=False,
    )
    payload["package_digest"] = digest_json(payload)
    return payload


def verify_package(receipt, *, weights=True):
    receipt = dict(receipt)
    claimed = receipt.pop("package_digest")
    if digest_json(receipt) != claimed:
        raise ValueError("Package receipt digest mismatch")
    receipt["package_digest"] = claimed
    checkpoint = Path(receipt["checkpoint"])
    if checkpoint.stat().st_size != receipt["checkpoint_bytes"]:
        raise ValueError("Checkpoint size changed")
    for name, expected in {**receipt["sidecars_sha256"], **receipt["tokenizer_sha256"]}.items():
        if sha256(checkpoint.parent / name) != expected:
            raise ValueError(f"Checkpoint sidecar changed: {name}")
    for name, expected in receipt["source_sha256"].items():
        if sha256(Path(receipt["source_snapshot"]) / name) != expected:
            raise ValueError(f"Training source changed: {name}")
    if weights and sha256(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash changed")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = build_package(args.checkpoint, args.source_snapshot)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(
        json.dumps(
            {k: receipt[k] for k in ("mode", "package_digest", "checkpoint_sha256", "tensor_count")}
        )
    )


if __name__ == "__main__":
    main()
