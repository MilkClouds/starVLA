# Checkpoint validation

Validated on 2026-09-25 against StarVLA base `4507931`, using the adapter in this
PR, `vla-eval==0.7.0`, Charliecloud, and an NVIDIA H100 80GB (driver 580.95.05).
The model environment used Python 3.11.15, PyTorch 2.14.0, torchvision 0.29.0,
Transformers 4.57.6, and NumPy 1.26.4.

These are representative smoke evaluations, not full benchmark score
reproductions or comparisons against the native evaluation clients.

## Results

| Check | Result | Artifact |
|---|---|---|
| Adapter contracts | 24 passed | `tests/test_vla_eval_adapter.py` |
| Full repository test suite | 69 passed, 2 subtests passed | `python -m pytest tests -q` |
| LIBERO spatial, GPU rendering, one episode | 1/1 success; 0 errors | [aggregate](validation/libero_gpu_smoke.json) |
| LIBERO spatial, CPU rendering, first two tasks × five episodes | 10/10 success; 0 errors | [aggregate](validation/libero_cpu.json) |
| RoboTwin 2.0, `grab_roller`, five seeds, GPU rendering | 5/5 success; 0 errors | [aggregate](validation/robotwin_gpu.json) |

Ruff and Black checks passed for the new Python files. The full test suite ran
in the uv model environment with pytest, matplotlib, pyzmq, wandb, and
DeepSpeed 0.16.9 added for existing tests. No system Python packages were changed.

Actual checkpoint statistics were also checked through `PolicyNormProcessor`:
LIBERO's six normalized action dimensions and unchanged gripper, and RoboTwin's
12 normalized joint dimensions and two binary grippers, followed by the
adapter's benchmark-facing conversion. Both passed.

The extended LIBERO GPU-rendering attempt aborted inside the simulator after
four successful episodes. It is not counted as a completed run. The same
10-episode protocol completed with CPU rendering (`--render cpu`); model
inference remained on the GPU. The single-episode GPU smoke completed normally.

## Checkpoints and environments

| Component | Revision |
|---|---|
| [StarVLA/Qwen3-VL-OFT-LIBERO-4in1](https://huggingface.co/StarVLA/Qwen3-VL-OFT-LIBERO-4in1) | `1947454be8c0f4de315f4cbde96b874819a6dab3`, `steps_50000_pytorch_model.pt` |
| [StarVLA/Qwen3-VL-OFT-RoboTwin2-All](https://huggingface.co/StarVLA/Qwen3-VL-OFT-RoboTwin2-All) | `727645249e6bfbff6870db4637e4b9d32e6be346`, `steps_140000_pytorch_model.pt` |
| [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |

Image manifests, verified against GHCR:

- `ghcr.io/allenai/vla-evaluation-harness/libero:0.7.0`:
  `sha256:9c88bfed3534405af175c9d174c4da1ea2533682bbb0a493c6444436eedf414f`
- `ghcr.io/allenai/vla-evaluation-harness/robotwin:0.7.0`:
  `sha256:93166b1741956ce638906ba2ba595e0d3c52e0b10ebe4dbf18045b4edc1f806f`

For exact reproduction, download the revisions above and pass the snapshot or
weight path using `--arg checkpoint=...`, and the base-VLM snapshot using
`--arg base_vlm=...`, to the model-server commands in [README.md](README.md).
The model uses bf16, SDPA, and the full checkpoint action horizon (8 / 50).

## Evaluation commands

With the corresponding model server running, the GPU LIBERO smoke uses the
unmodified example config:

```bash
uv run --extra evaluation vla-eval run --runtime charliecloud --yes \
  --config examples/vla_eval/benchmarks/libero_smoke.yaml --record-video
```

The extended LIBERO run uses suite `libero_spatial`, seed 7, and 10 settling steps:

```bash
uv run --extra evaluation vla-eval run --runtime charliecloud --yes \
  --config examples/vla_eval/benchmarks/libero_smoke.yaml --render cpu \
  --benchmark-field max_tasks=2 --benchmark-field episodes_per_task=5 \
  --output-dir results/vla_eval/libero_validation --record-video
```

RoboTwin uses `grab_roller`, `demo_clean`, seed 0, and five distinct environment
seeds 100000–100004 selected by `test_num=5`. Expert filtering is disabled by the smoke config;
this also uses the harness's generic task instruction instead of the official
expert-generated instructions. It must not be compared to official scores.

```bash
uv run --extra evaluation vla-eval run --runtime charliecloud --yes \
  --config examples/vla_eval/benchmarks/robotwin_smoke.yaml --param test_num=5 \
  --output-dir results/vla_eval/robotwin_validation --record-video
```

Aggregate JSON is checked in; per-step SQLite recordings, videos, and full logs
are retained with the local evaluation artifacts and are not part of the PR.
