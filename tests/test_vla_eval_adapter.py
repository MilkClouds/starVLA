from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("vla_eval")

from vla_eval.cli.config_loader import load_config
from vla_eval.registry import resolve_import_string

from deployment.vla_eval import model_server as adapter


class _FakePolicy:
    metadata: ClassVar = {
        "action_chunk_size": 2,
        "default_unnorm_key": "test",
    }

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls = []

    def predict_action(self, **kwargs):
        self.calls.append(kwargs)
        batch = len(kwargs["examples"])
        action_dim = 7 if len(kwargs["examples"][0]["image"]) == 2 else 14
        actions = np.zeros((batch, 2, action_dim), dtype=np.float32)
        if action_dim == 7:
            actions[..., 6] = 1.0
        else:
            actions[..., 6] = 1.0
            actions[..., 12] = 12.0
            actions[..., 13] = 13.0
        return {"actions": actions}


@pytest.fixture
def fake_policy(monkeypatch):
    monkeypatch.setattr(adapter, "resolve_checkpoint", lambda checkpoint: "/tmp/run/checkpoints/model.pt")
    monkeypatch.setattr(adapter, "_build_policy", _FakePolicy)


def test_libero_contract_orders_cameras_and_converts_gripper(fake_policy):
    server = adapter.StarVLAHarnessServer(
        "unused",
        base_vlm="Qwen/Qwen3-VL-4B-Instruct",
        image_size=[4, 5],
        image_keys=["agentview", "wrist"],
        gripper_transform="open01_to_close_positive",
        action_profile="libero",
    )
    obs = {
        "images": {
            "wrist": np.full((2, 3, 3), 2, dtype=np.uint8),
            "agentview": np.full((2, 3, 3), 1, dtype=np.uint8),
        },
        "task_description": "pick",
    }

    result = server.predict_batch([obs], [SimpleNamespace()])[0]["actions"]
    example = server._policy.calls[0]["examples"][0]

    assert [np.asarray(image)[0, 0, 0] for image in example["image"]] == [1, 2]
    assert [np.asarray(image).shape for image in example["image"]] == [(4, 5, 3), (4, 5, 3)]
    assert example["lang"] == "pick"
    assert "state" not in example
    assert server._policy.init_kwargs["config_overrides"] == ["framework.qwenvl.base_vlm=Qwen/Qwen3-VL-4B-Instruct"]
    np.testing.assert_array_equal(result[:, 6], -1.0)
    assert server.chunk_size == 2


def test_robotwin_contract_reorders_actions_and_omits_state(fake_policy):
    indices = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
    server = adapter.StarVLAHarnessServer(
        "unused",
        image_keys=["head_camera", "left_camera", "right_camera"],
        action_indices=indices,
        action_profile="robotwin",
    )
    obs = {
        "images": {
            "right_camera": np.full((2, 2, 3), 2, dtype=np.uint8),
            "head_camera": np.zeros((2, 2, 3), dtype=np.uint8),
            "left_camera": np.ones((2, 2, 3), dtype=np.uint8),
        },
        "joint_state": np.arange(14, dtype=np.float32),
        "task_description": "grab roller",
    }

    result = server.predict_batch([obs], [SimpleNamespace()])[0]["actions"]
    example = server._policy.calls[0]["examples"][0]

    assert [np.asarray(image)[0, 0, 0] for image in example["image"]] == [0, 1, 2]
    assert "state" not in example
    np.testing.assert_array_equal(result[0], np.array([0, 0, 0, 0, 0, 0, 12, 1, 0, 0, 0, 0, 0, 13]))


def test_configured_missing_camera_fails_fast(fake_policy):
    server = adapter.StarVLAHarnessServer("unused", image_keys=["agentview", "wrist"])
    with pytest.raises(KeyError, match="wrist"):
        server._make_example(
            {
                "images": {"agentview": np.zeros((2, 2, 3), dtype=np.uint8)},
                "task_description": "test",
            }
        )


def test_explicit_state_contract(fake_policy):
    server = adapter.StarVLAHarnessServer("unused", image_keys=["agentview"], state_key="states")
    example = server._make_example(
        {
            "images": {"agentview": np.zeros((2, 2, 3), dtype=np.uint8)},
            "states": np.arange(7, dtype=np.float32),
        }
    )
    assert example["state"].shape == (1, 7)


def test_shipped_configs_resolve():
    root = Path(__file__).resolve().parents[1]
    benchmark_paths = [
        root / "examples/vla_eval/benchmarks/libero_smoke.yaml",
        root / "examples/vla_eval/benchmarks/robotwin_smoke.yaml",
    ]
    model_paths = [
        root / "examples/vla_eval/model_servers/libero_qwen3_oft.yaml",
        root / "examples/vla_eval/model_servers/robotwin_qwen3_oft.yaml",
    ]

    for path in benchmark_paths:
        config = load_config(str(path))
        for benchmark in config["benchmarks"]:
            resolve_import_string(benchmark["benchmark"])

    for path in model_paths:
        config = load_config(str(path))
        assert (root / config["script"]).is_file()
        assert config["args"]["base_vlm"] == "Qwen/Qwen3-VL-4B-Instruct"


def test_checkpoint_directory_selects_latest_numeric_step(tmp_path):
    (tmp_path / "config.yaml").write_text("{}")
    (tmp_path / "dataset_statistics.json").write_text("{}")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "steps_9000.pt").touch()
    latest = checkpoint_dir / "steps_10000.safetensors"
    latest.touch()

    assert adapter.resolve_checkpoint(str(tmp_path)) == str(latest)


def test_checkpoint_symlink_preserves_colocated_metadata(tmp_path):
    root = tmp_path / "snapshot"
    (root / "checkpoints").mkdir(parents=True)
    (root / "config.yaml").write_text("{}")
    (root / "dataset_statistics.json").write_text("{}")
    blob = tmp_path / "blob-without-extension"
    blob.touch()
    checkpoint = root / "checkpoints/model.pt"
    checkpoint.symlink_to(blob)

    assert adapter.resolve_checkpoint(str(checkpoint)) == str(checkpoint)


@pytest.mark.parametrize("missing", ["config.yaml", "dataset_statistics.json"])
def test_checkpoint_requires_colocated_metadata(tmp_path, missing):
    (tmp_path / "checkpoints").mkdir()
    checkpoint = tmp_path / "checkpoints/model.pt"
    checkpoint.touch()
    for name in {"config.yaml", "dataset_statistics.json"} - {missing}:
        (tmp_path / name).write_text("{}")

    with pytest.raises(FileNotFoundError, match=missing):
        adapter.resolve_checkpoint(str(checkpoint))


def test_hub_checkpoint_download_includes_config_and_stats(tmp_path, monkeypatch):
    import huggingface_hub

    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "config.yaml").write_text("{}")
    (tmp_path / "dataset_statistics.json").write_text("{}")
    checkpoint = tmp_path / "checkpoints/model.safetensors"
    checkpoint.touch()

    def download(repo_id, *, allow_patterns):
        assert repo_id == "StarVLA/example"
        assert {"config.yaml", "dataset_statistics.json"} <= set(allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert adapter.resolve_checkpoint("StarVLA/example") == str(checkpoint)


@pytest.mark.parametrize("unnorm_key", [None, "alternate"])
def test_checkpoint_statistics_key_forwarded_to_wrapper_and_inference(fake_policy, unnorm_key):
    server = adapter.StarVLAHarnessServer("unused", unnorm_key=unnorm_key, image_keys=["agentview", "wrist"])
    obs = {"images": {key: np.zeros((2, 2, 3), dtype=np.uint8) for key in ["wrist", "agentview"]}}
    server.predict_batch([obs], [SimpleNamespace()])

    assert server._policy.init_kwargs["unnorm_key"] == unnorm_key
    assert server._policy.calls[0]["unnorm_key"] == (unnorm_key or "test")


def test_ambiguous_checkpoint_statistics_fail_at_startup(fake_policy, monkeypatch):
    monkeypatch.setattr(
        _FakePolicy,
        "metadata",
        {
            "action_chunk_size": 2,
            "available_unnorm_keys": ["first", "second"],
            "default_unnorm_key": None,
        },
    )
    with pytest.raises(ValueError, match="multiple statistics keys"):
        adapter.StarVLAHarnessServer("unused")
    server = adapter.StarVLAHarnessServer("unused", unnorm_key="second")
    assert server._unnorm_key == "second"
    with pytest.raises(ValueError, match="Unknown unnorm_key"):
        adapter.StarVLAHarnessServer("unused", unnorm_key="missing")


def test_gripper_threshold_and_joint_values_are_preserved():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    actions[:, -1] = [0.0, 0.5, 1.0]
    original = actions.copy()
    result = adapter.postprocess_actions(actions, gripper_transform="open01_to_close_positive")
    np.testing.assert_array_equal(result[:, -1], [1.0, 1.0, -1.0])
    np.testing.assert_array_equal(result[:, :6], original[:, :6])
    np.testing.assert_array_equal(actions, original)


@pytest.mark.parametrize("chunk_size", [0, -1, 3])
def test_invalid_chunk_size_rejected(fake_policy, chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        adapter.StarVLAHarnessServer("unused", chunk_size=chunk_size)


def test_batch_keeps_observations_and_actions_separate(fake_policy):
    server = adapter.StarVLAHarnessServer("unused", image_keys=["agentview", "wrist"], max_batch_size=2)
    observations = [
        {
            "task_description": task,
            "images": {key: np.zeros((2, 2, 3), dtype=np.uint8) for key in ["agentview", "wrist"]},
        }
        for task in ["pick", "place"]
    ]
    results = server.predict_batch(observations, [SimpleNamespace(), SimpleNamespace()])
    assert [example["lang"] for example in server._policy.calls[0]["examples"]] == ["pick", "place"]
    assert len(results) == 2
    assert all(result["actions"].shape == (2, 7) for result in results)


@pytest.mark.parametrize("shape", [(2, 7), (2, 2, 7), (1, 1, 7)])
def test_malformed_policy_output_rejected(fake_policy, monkeypatch, shape):
    server = adapter.StarVLAHarnessServer("unused", image_keys=["agentview", "wrist"])
    monkeypatch.setattr(server._policy, "predict_action", lambda **kwargs: {"actions": np.zeros(shape)})
    obs = {"images": {key: np.zeros((2, 2, 3), dtype=np.uint8) for key in ["wrist", "agentview"]}}
    with pytest.raises(ValueError, match="actions"):
        server.predict_batch([obs], [SimpleNamespace()])


def test_chunk_buffer_resets_between_episodes(fake_policy):
    import asyncio

    from vla_eval.model_servers.base import SessionContext

    async def exercise():
        server = adapter.StarVLAHarnessServer("unused", image_keys=["agentview", "wrist"])
        obs = {"images": {key: np.zeros((2, 2, 3), dtype=np.uint8) for key in ["wrist", "agentview"]}}
        sent = []

        async def send(action):
            sent.append(action["actions"])

        ctx = SessionContext(session_id="session", episode_id="first")
        ctx._send_action_fn = send
        await server.on_episode_start({}, ctx)
        for _ in range(3):
            await server.on_observation(obs, ctx)
            ctx._increment_step()
        assert len(server._policy.calls) == 2
        assert len(sent) == 3
        assert all(action.shape == (7,) for action in sent)

        next_ctx = SessionContext(session_id="session", episode_id="second")
        next_ctx._send_action_fn = send
        await server.on_episode_start({}, next_ctx)
        await server.on_observation(obs, next_ctx)
        assert len(server._policy.calls) == 3

    asyncio.run(exercise())


@pytest.mark.parametrize("profile", ["libero", "robotwin"])
def test_shipped_model_configs_match_native_contract(fake_policy, profile):
    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / f"examples/vla_eval/model_servers/{profile}_qwen3_oft.yaml"))
    server = adapter.StarVLAHarnessServer(**config["args"])
    keys = ["agentview", "wrist"] if profile == "libero" else ["head_camera", "left_camera", "right_camera"]
    obs = {"images": {key: np.full((3, 4, 3), index, dtype=np.uint8) for index, key in reversed(list(enumerate(keys)))}}
    actions = server.predict_batch([obs], [SimpleNamespace()])[0]["actions"]
    example = server._policy.calls[0]["examples"][0]
    assert [np.asarray(image)[0, 0, 0] for image in example["image"]] == list(range(len(keys)))
    assert all(image.size == (224, 224) for image in example["image"])
    assert "state" not in example
    assert server.get_observation_params() == ({"send_wrist_image": True} if profile == "libero" else {})
    if profile == "libero":
        np.testing.assert_array_equal(actions[:, 6], -1.0)
    else:
        np.testing.assert_array_equal(actions[0], [0, 0, 0, 0, 0, 0, 12, 1, 0, 0, 0, 0, 0, 13])
        assert server._policy.calls[0]["unnorm_key"] == "new_embodiment"
