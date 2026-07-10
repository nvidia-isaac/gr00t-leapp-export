# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert a LEAPP-exported ``exported_gr00t.yaml`` into the Isaac/ROS-compatible
``exported_gr00t_isaac.yaml`` variant by adding ``source: <name>`` to every
model input. The Isaac runtime uses these ``source`` fields to bind ROS topics
to graph inputs; without them, inference cannot run end-to-end through the ROS
pipeline.

Usage:
    python postprocess_isaac_yaml.py <exported_model_dir>
    python postprocess_isaac_yaml.py <input.yaml> -o <output.yaml>
"""

import argparse
import os
import sys

import yaml


def add_source_to_inputs(config: dict) -> dict:
    models = config.get("models", {})
    for node_name, node in models.items():
        for inp in node.get("inputs", []) or []:
            inp["source"] = inp["name"]
    return config


def convert(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        config = yaml.safe_load(f)

    config = add_source_to_inputs(config)

    with open(output_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    print(f"Wrote: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to exported_gr00t.yaml OR the export dir containing it.")
    p.add_argument("-o", "--output", default=None,
                   help="Output yaml path. Defaults to <input_dir>/exported_gr00t_isaac.yaml.")
    args = p.parse_args()

    if os.path.isdir(args.path):
        # Pick the single non-_isaac yaml in the directory.
        candidates = [
            f for f in os.listdir(args.path)
            if f.endswith(".yaml") and not f.endswith("_isaac.yaml")
        ]
        if len(candidates) != 1:
            print(
                f"ERROR: expected exactly one *.yaml (excluding *_isaac.yaml) in {args.path}, "
                f"found {candidates}. Pass the file path directly.",
                file=sys.stderr,
            )
            sys.exit(1)
        input_path = os.path.join(args.path, candidates[0])
    else:
        input_path = args.path

    if not os.path.isfile(input_path):
        print(f"ERROR: not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or os.path.join(
        os.path.dirname(input_path), "exported_gr00t_isaac.yaml"
    )
    convert(input_path, output_path)


if __name__ == "__main__":
    main()
