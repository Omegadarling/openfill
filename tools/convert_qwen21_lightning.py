#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert the pinned LightX2V transformer to BF16 using an existing NumPy environment."""

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path

import numpy as np


SOURCE_REVISION = "682b2529ffb5ae9ba39ceb81a31ce99e602da2ee"
SOURCE_SHA256 = "4952ed381a112fa4725d4aa679050179482267b7e4c308a067b5ca0ca8915248"
SOURCE_SIZE = 28_460_533_992
CHUNK_SIZE = 8 * 1024 * 1024
SQUARE = [4096, 4096]
INPUT_MLP = [12288, 4096]


def expected_shapes():
    shapes = {
        "img_in.weight": [4096, 64],
        "modulation.1.weight": [16384, 4096],
        "norm_out.linear.weight": SQUARE,
        "proj_out.weight": [64, 4096],
        "time_text_embed.timestep_embedder.linear_1.weight": [4096, 256],
        "time_text_embed.timestep_embedder.linear_2.weight": SQUARE,
        "txt_in.in_layer.weight": SQUARE,
        "txt_in.out_layer.weight": SQUARE,
        "txt_in.text_norm.weight": [4096],
    }
    block = {
        "attn.norm_k.weight": [128],
        "attn.norm_q.weight": [128],
        "attn.to_k.weight": SQUARE,
        "attn.to_out.0.weight": SQUARE,
        "attn.to_q.weight": SQUARE,
        "attn.to_v.weight": SQUARE,
        "img_mlp.gate_layer.weight": INPUT_MLP,
        "img_mlp.out.weight": [4096, 12288],
        "img_mlp.proj.weight": INPUT_MLP,
    }
    for index in range(32):
        shapes.update({f"transformer_blocks.{index}.{name}": shape for name, shape in block.items()})
    return shapes


def read_header(source):
    length_bytes = source.read(8)
    length = struct.unpack("<Q", length_bytes)[0]
    if length > 1024 * 1024:
        raise ValueError("Unexpected safetensors header size")
    header_bytes = source.read(length)
    header = json.loads(header_bytes)
    if header.pop("__metadata__") != {"format": "pt"}:
        raise ValueError("Unexpected source metadata")
    shapes = expected_shapes()
    if header.keys() != shapes.keys():
        raise ValueError("Source tensor names differ from the native Qwen-Image 2.1 architecture")

    output = {
        "__metadata__": {
            "format": "pt",
            "source_model": "lightx2v/Qwen-Image-2.1-Lightning-v2-RL",
            "source_revision": SOURCE_REVISION,
            "source_sha256": SOURCE_SHA256,
            "conversion": "FP32 to BF16 (round-to-nearest-even), tensor names unchanged",
        }
    }
    cursor = 0
    for name, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        end = cursor + 4 * math.prod(shapes[name])
        if entry != {"dtype": "F32", "shape": shapes[name], "data_offsets": [cursor, end]}:
            raise ValueError(f"Unexpected source dtype, shape, or offset for {name}")
        output[name] = {"dtype": "BF16", "shape": shapes[name], "data_offsets": [cursor // 2, end // 2]}
        cursor = end
    if 8 + length + cursor != SOURCE_SIZE:
        raise ValueError("Unexpected source data size")
    return length_bytes + header_bytes, output, cursor


def to_bf16(chunk):
    bits = np.frombuffer(chunk, dtype="<u4")
    if np.any((bits & 0x7F800000) == 0x7F800000):
        raise ValueError("Source checkpoint contains a non-finite weight")
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype("<u2").tobytes()


def convert(source_path, output_path):
    if output_path.exists():
        raise FileExistsError(output_path)
    temporary_path = None
    try:
        with source_path.open("rb") as source:
            if os.fstat(source.fileno()).st_size != SOURCE_SIZE:
                raise ValueError(f"Expected {SOURCE_SIZE:,} bytes for the pinned FP32 checkpoint")
            raw_header, output_header, remaining = read_header(source)
            source_hash = hashlib.sha256(raw_header)
            output_hash = hashlib.sha256()
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".partial", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                header = json.dumps(output_header, separators=(",", ":")).encode()
                header += b" " * (-len(header) % 8)
                serialized_header = struct.pack("<Q", len(header)) + header
                temporary.write(serialized_header)
                output_hash.update(serialized_header)
                while remaining:
                    size = min(CHUNK_SIZE, remaining)
                    chunk = source.read(size)
                    if len(chunk) != size:
                        raise ValueError("Source file ended mid-conversion")
                    source_hash.update(chunk)
                    data = to_bf16(chunk)
                    temporary.write(data)
                    output_hash.update(data)
                    remaining -= size
                if source.read(1) or source_hash.hexdigest() != SOURCE_SHA256:
                    raise ValueError("Source SHA-256 does not match the pinned publisher checkpoint")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.link(temporary_path, output_path)
        print(json.dumps({"path": str(output_path), "bytes": output_path.stat().st_size, "sha256": output_hash.hexdigest()}))
    finally:
        if temporary_path is not None:
            temporary_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pinned LightX2V diffusion_pytorch_model.safetensors")
    parser.add_argument("output", type=Path, help="New BF16 safetensors path in an existing directory")
    arguments = parser.parse_args()
    convert(arguments.source, arguments.output)


if __name__ == "__main__":
    main()
