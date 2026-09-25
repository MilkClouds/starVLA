# vla-eval

Optional LIBERO and RoboTwin 2.0 evaluation through `vla-eval`. The adapter reuses
`PolicyServerWrapper` for checkpoint loading and action unnormalization.

Install the frontend from the repository root:

```bash
uv sync --python 3.11 --extra evaluation
```

Start the model server on a GPU, then run the benchmark in another terminal:

```bash
uv run --extra evaluation vla-eval serve \
  --config examples/vla_eval/model_servers/libero_qwen3_oft.yaml
uv run --extra evaluation vla-eval run --runtime charliecloud --yes \
  --config examples/vla_eval/benchmarks/libero_smoke.yaml
```

Install [Charliecloud](https://hpc.github.io/charliecloud/) and its NVIDIA driver
injection dependencies first. `serve` creates a separate uv environment for the
model dependencies. For RoboTwin, substitute `robotwin_qwen3_oft.yaml` and
`robotwin_smoke.yaml` in the commands above.

The model configs accept a Hugging Face repository ID, local run directory, or
weight file via `--arg checkpoint=...`. A run needs `config.yaml`,
`dataset_statistics.json`, and `checkpoints/*.pt` or `*.safetensors`. Directories
select the highest numeric step; use a specific weight path to pin a checkpoint.
Set `--arg unnorm_key=...` if checkpoint statistics have no unambiguous default.

| Profile | Camera order | Action conversion |
|---|---|---|
| LIBERO | `agentview`, `wrist` | gripper: `1 - 2 * (x > 0.5)` |
| RoboTwin | `head_camera`, `left_camera`, `right_camera` | left arm/gripper, then right arm/gripper |

Both supplied OFT configs omit state. Images are resized to 224 × 224 using PIL
bilinear for LIBERO and OpenCV area for RoboTwin, matching the native clients.
Actions use the checkpoint's statistics and chunk horizon; episode boundaries
clear the harness buffer.

The configs run smoke evaluations, not full-suite score reproductions. RoboTwin
skips expert filtering and uses the harness's generic task instruction. Results
are saved to the configured `output_dir`.

Run the adapter contract tests with:

```bash
uv run --extra dev --extra evaluation pytest tests/test_vla_eval_adapter.py
```
