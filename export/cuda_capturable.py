# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite the exported ``backbone.onnx`` into a CUDA-graph-capturable variant.

The GR00T N1.7 backbone contains a handful of ``NonZero`` ops (the ONNX form of
``Tensor.masked_scatter_()`` over masks derived from ``input_ids`` / ``attention_mask``).
``NonZero``'s output shape is *data-dependent*, so TensorRT must read the element count
device->host mid-``enqueueV3`` — which is illegal inside CUDA-graph capture. For a fixed-prompt
export the prompt is baked inside ``preprocess_video`` (its only runtime input is the image), so
those tensors are invariant and every ``NonZero`` output is a *constant*. This step derives those
constants and bakes them in as initializers, dropping the ``NonZero`` nodes; the now-orphaned mask
subgraphs are dead and pruned by TensorRT at build time.
"""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, utils


def _find_nonzero_outputs(graph) -> list[str]:
    """Return the single output name of every ``NonZero`` node in *graph*."""
    return [
        node.output[0]
        for node in graph.node
        if node.op_type == "NonZero" and len(node.output) == 1
    ]


def _ancestor_graph_inputs(graph, targets: list[str]) -> list[str]:
    """Graph inputs that *targets* transitively depend on (backward reachability)."""
    producer = {out: node for node in graph.node for out in node.output}
    graph_input_names = {i.name for i in graph.input}
    reachable_inputs: set[str] = set()
    seen: set[str] = set()
    stack = list(targets)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in graph_input_names:
            reachable_inputs.add(name)
        node = producer.get(name)
        if node is not None:
            stack.extend(node.input)
    # Preserve graph input order for deterministic output.
    return [i.name for i in graph.input if i.name in reachable_inputs]


def _run_preprocess_video(path: str, seed: int) -> dict[str, np.ndarray]:
    """Run ``preprocess_video.onnx`` on a random image and return its outputs by name."""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    # First dim is batch; keep it at 1. Remaining dims are the fixed image geometry.
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    rng = np.random.default_rng(seed)
    dtype = np.float32 if "float" in inp.type else np.int64
    image = (rng.random(shape, np.float32) * 255.0).astype(dtype)
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {inp.name: image})))


def _match_feed(needed_inputs, pv_outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map backbone subgraph inputs to preprocess_video outputs by name suffix, casting dtypes."""
    feed = {}
    for vi in needed_inputs:
        np_dtype = helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
        match = next((k for k in pv_outputs if vi.name.endswith(k)), None)
        if match is None:
            raise SystemExit(
                f"make_backbone_capturable: backbone input '{vi.name}' has no matching "
                f"preprocess_video output (available: {sorted(pv_outputs)})."
            )
        feed[vi.name] = pv_outputs[match].astype(np_dtype)
    return feed


def make_cuda_capturable(model_dir: str) -> None:
    """Rewrite ``<model_dir>/backbone.onnx`` in place into a CUDA-graph-capturable form.

    No-op (with a message) if the backbone has no ``NonZero`` ops.
    """
    backbone_path = os.path.join(model_dir, "backbone.onnx")
    pv_path = os.path.join(model_dir, "preprocess_video.onnx")

    # 1. Prompt-invariance guard: the bake is only valid if the tokens do not depend on the image.
    #    Run preprocess_video on two different images and require identical tokens.
    o1 = _run_preprocess_video(pv_path, seed=0)
    o2 = _run_preprocess_video(pv_path, seed=1)
    for key in ("input_ids", "attention_mask"):
        if key in o1 and not np.array_equal(o1[key], o2.get(key)):
            raise SystemExit(
                f"make_backbone_capturable: preprocess_video '{key}' depends on the input image, so "
                "the prompt is not fixed and the NonZero outputs are not constant. Refusing to patch "
                "— this backbone cannot be made CUDA-graph-capturable."
            )

    # 2. Discover the NonZero outputs; nothing to do if there are none.
    model = onnx.load(backbone_path, load_external_data=False)  # weights stay external
    graph = model.graph
    nonzero_outputs = _find_nonzero_outputs(graph)
    if not nonzero_outputs:
        print("make_backbone_capturable: no NonZero ops found; backbone already capturable.")
        return

    # 3. Extract the (vision-free) subgraph that computes the NonZero outputs, run it, read values.
    #    Register the internal NonZero tensors as graph outputs so the Extractor can target them.
    existing = {o.name for o in graph.output}
    for nz in nonzero_outputs:
        if nz not in existing:
            graph.output.append(helper.make_empty_tensor_value_info(nz))
    sub_inputs = _ancestor_graph_inputs(graph, nonzero_outputs)
    subgraph = utils.Extractor(model).extract_model(sub_inputs, nonzero_outputs)

    feed = _match_feed(subgraph.graph.input, o1)

    # onnxruntime refuses external-data paths that escape the model dir, so run the subgraph in a
    # temp dir on the same filesystem with the weights hard-linked in.
    weights = os.path.realpath(os.path.join(model_dir, "backbone.onnx.data"))
    with tempfile.TemporaryDirectory(dir=model_dir) as tmp:
        os.link(weights, os.path.join(tmp, "backbone.onnx.data"))
        sub_path = os.path.join(tmp, "mask_subgraph.onnx")
        onnx.save(subgraph, sub_path)
        sess = ort.InferenceSession(sub_path, providers=["CPUExecutionProvider"])
        values = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))

    # 4. Drop the NonZero nodes and replace each output with an identically-named constant
    #    initializer, so every downstream consumer reads the static value.
    fresh_model = onnx.load(backbone_path, load_external_data=False)
    fresh_graph = fresh_model.graph
    kept = [
        node
        for node in fresh_graph.node
        if not (node.op_type == "NonZero" and len(node.output) == 1
                and node.output[0] in set(nonzero_outputs))
    ]
    del fresh_graph.node[:]
    fresh_graph.node.extend(kept)
    for nz in nonzero_outputs:
        fresh_graph.initializer.append(
            numpy_helper.from_array(values[nz].astype(np.int64), name=nz)
        )

    onnx.save(fresh_model, backbone_path)
    print(
        f"make_cuda_capturable: patched {backbone_path} "
        f"(removed {len(nonzero_outputs)} NonZero ops, baked {len(nonzero_outputs)} constants)."
    )
