# vla-eval integration

An optional [vla-eval](https://github.com/allenai/vla-evaluation-harness) frontend
for StarVLA, with LIBERO and RoboTwin 2.0 examples. The adapter uses the existing
`PolicyServerWrapper` for checkpoint loading and training-time action
unnormalization. Existing benchmark evaluation scripts remain available.

## Setup

From the repository root, install the optional frontend and test dependencies:

```bash
uv sync --python 3.11 --extra dev --extra evaluation
```

`vla-eval serve` installs the model script's PEP 723 dependencies into a separate
uv environment and imports model code from this checkout. The supplied model
configs replace the released checkpoints' machine-local base-VLM paths with
`Qwen/Qwen3-VL-4B-Instruct`.

## Evaluate a released checkpoint

Start the model server on a GPU:

```bash
uv run --extra evaluation vla-eval serve \
  --config examples/vla_eval/model_servers/libero_qwen3_oft.yaml
```

In another terminal, run one episode and record its results and video:

```bash
uv run --extra evaluation vla-eval run --runtime charliecloud --yes \
  --config examples/vla_eval/benchmarks/libero_smoke.yaml --record-video
```

Install [Charliecloud](https://hpc.github.io/charliecloud/) and its NVIDIA driver
injection dependencies on the host first. On hosts where Docker is permitted,
`--runtime docker` uses the same benchmark images. Images are tagged `0.7.0`;
record their resolved digests when reporting results.

For RoboTwin 2.0, use `robotwin_qwen3_oft.yaml` and `robotwin_smoke.yaml` in the
same directories. The RoboTwin smoke config skips the expert solvability check;
it is a plumbing check, not the official score protocol.

To use an existing checkpoint, pass its run directory or weight file:

```bash
uv run --extra evaluation vla-eval serve \
  --config examples/vla_eval/model_servers/robotwin_qwen3_oft.yaml \
  --arg checkpoint=/absolute/path/to/run/checkpoints/steps_140000_pytorch_model.pt
```

The run directory must contain `config.yaml`, `dataset_statistics.json`, and
`checkpoints/*.pt` or `checkpoints/*.safetensors`. A directory or Hugging Face
repository ID selects the highest numeric checkpoint step. For a reproducible
run, use a specific snapshot/weight path. Cache symlinks are preserved so the
wrapper can locate the accompanying metadata. Multi-key statistics require an
explicit `--arg unnorm_key=...` unless the wrapper supplies a default.

Use `--address 127.0.0.1:8001` on `serve` and
`--server-url ws://127.0.0.1:8001` on `run` to change the port. To expand a smoke
run, LIBERO accepts `--benchmark-field episodes_per_task=5` and
`--benchmark-field max_tasks=2`. For RoboTwin, use `--param test_num=5` to
evaluate five distinct seeds. Results, per-step recordings, and optional videos
are written under the benchmark config's `output_dir`.

## Adapter contract

| Profile | Camera order | Model state | Action conversion |
|---|---|---|---|
| LIBERO | `agentview`, `wrist` | omitted | open `[0, 1]` to close-positive `[-1, 1]`, threshold `> 0.5` |
| RoboTwin 2.0 | `head_camera`, `left_camera`, `right_camera` | omitted for absolute-action OFT | training order to left arm/gripper, right arm/gripper qpos |

LIBERO uses PIL bilinear resizing; RoboTwin uses OpenCV area resizing, both to
224 × 224, matching the native clients. Images arrive in the harness benchmark's
orientation. The adapter does not apply another flip or rotation.

Action unnormalization happens once, inside `PolicyServerWrapper`, before the
adapter converts the gripper or reorders dimensions. `chunk_size` defaults to
the checkpoint horizon (8 for the supplied LIBERO checkpoint, 50 for RoboTwin).
The harness buffers each chunk and clears it at episode boundaries.
`max_batch_size` defaults to 1; increasing it enables model-side batching across
sessions for policies that support batched `predict_action`.

## Validation

```bash
uv run --extra dev --extra evaluation pytest tests/test_vla_eval_adapter.py
```

Contract tests cover camera order, resizing, action conversion, batching,
checkpoint metadata paths, and statistics-key selection. Representative GPU
simulator runs and their exact provenance are recorded in [VALIDATION.md](VALIDATION.md).
Smoke completion checks the integration; it does not establish full-suite
score parity with the original evaluation scripts. Additional benchmarks and
checkpoint reproductions are outside this initial integration.
