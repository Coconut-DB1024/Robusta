from __future__ import annotations

import hashlib
import struct
from typing import Iterable, List, Sequence, Tuple


_MASK64 = (1 << 64) - 1
_EMPTY = _MASK64
_MINHASH_SEED = 0x243F6A8885A308D3
_RANK_SEED = 0x13198A2E03707344
_DENSIFY_SEED = 0xA4093822299F31D0


def _mix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def one_permutation_minhash(
    shingle_hashes: Sequence[int],
    signature_size: int,
) -> Tuple[int, ...]:
    signature_size = int(signature_size)
    if signature_size <= 0:
        raise ValueError("signature_size must be greater than zero")
    if not shingle_hashes:
        return tuple(0 for _ in range(signature_size))

    signature = [_EMPTY] * signature_size
    for raw_hash in shingle_hashes:
        permuted = _mix64(int(raw_hash) ^ _MINHASH_SEED)
        bucket = permuted % signature_size
        rank = _mix64(permuted ^ _RANK_SEED)
        if rank < signature[bucket]:
            signature[bucket] = rank

    nonempty = [index for index, value in enumerate(signature) if value != _EMPTY]
    if not nonempty:
        return tuple(0 for _ in range(signature_size))

    for index, value in enumerate(signature):
        if value != _EMPTY:
            continue
        for distance in range(1, signature_size + 1):
            donor = (index + distance) % signature_size
            donor_value = signature[donor]
            if donor_value == _EMPTY:
                continue
            signature[index] = _mix64(
                donor_value
                ^ _DENSIFY_SEED
                ^ _mix64((index << 32) | distance)
            )
            break

    return tuple(signature)


def lsh_band_keys(signature: Sequence[int], rows_per_band: int) -> Tuple[bytes, ...]:
    rows_per_band = int(rows_per_band)
    if rows_per_band <= 0 or len(signature) % rows_per_band != 0:
        raise ValueError("rows_per_band must divide the signature length")

    keys = []
    for band_index, start in enumerate(range(0, len(signature), rows_per_band)):
        rows = signature[start : start + rows_per_band]
        payload = struct.pack("<I", band_index) + struct.pack(
            f"<{rows_per_band}Q", *rows
        )
        keys.append(
            hashlib.blake2b(
                payload,
                digest_size=12,
                person=b"s3-lsh-v1",
            ).digest()
        )
    return tuple(keys)


def score_bucket(score: float) -> str:
    value = float(score)
    if value <= 0.0:
        return "0%"
    if value >= 1.0:
        return "100%"
    decile = int(value * 10.0)
    if decile <= 0:
        return "(0%, 10%)"
    lower = decile * 10
    return f"[{lower}%, {lower + 10}%)"


BUCKET_ORDER = [
    "100%",
    "[90%, 100%)",
    "[80%, 90%)",
    "[70%, 80%)",
    "[60%, 70%)",
    "[50%, 60%)",
    "[40%, 50%)",
    "[30%, 40%)",
    "[20%, 30%)",
    "[10%, 20%)",
    "(0%, 10%)",
    "0%",
]
