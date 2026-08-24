"""Tests for the config-driven metriq_qudits.cli argument parsing and orchestration.

``main`` is exercised with the benchmark's ``run`` monkeypatched to a recorder so
the expensive compile/pulse/simulate pipeline never runs; only the CLI's own
handling (config loading, device selection, per-dimension iteration) is tested.
"""

import json

import pytest

from metriq_qudits import cli


def _write_config(tmp_path, **overrides):
    config = {"benchmark_name": "quantum_volume", "dimensions": [4, 6], "n_unitaries": 3}
    config.update(overrides)
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(config))
    return str(path)


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

def test_parse_args_defaults(tmp_path):
    cfg = _write_config(tmp_path)
    args = cli._parse_args([cfg])
    assert args.config == cfg
    assert args.device is None
    assert args.overwrite is False
    assert args.output_dir is None


def test_parse_args_requires_config():
    with pytest.raises(SystemExit):
        cli._parse_args([])


def test_parse_args_flags(tmp_path):
    cfg = _write_config(tmp_path)
    args = cli._parse_args(
        [cfg, "--device", "dev.json", "--n-jobs", "4", "--overwrite", "--output-dir", "/tmp/o"]
    )
    assert args.device == "dev.json"
    assert args.n_jobs == 4
    assert args.overwrite is True
    assert args.output_dir == "/tmp/o"


def test_parse_args_rejects_non_integer_n_jobs(tmp_path):
    cfg = _write_config(tmp_path)
    with pytest.raises(SystemExit):
        cli._parse_args([cfg, "--n-jobs", "eight"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@pytest.fixture
def recorded_runs(monkeypatch):
    """Replace the benchmark's run with a recorder so the pipeline never runs."""
    from metriq_qudits.benchmarks.quantum_volume import QuantumVolume

    calls = []

    def recorder(self, config, device):
        calls.append((config, device))

    monkeypatch.setattr(QuantumVolume, "run", recorder)
    return calls


def test_main_runs_once_per_dimension(recorded_runs, tmp_path, capsys):
    cfg = _write_config(tmp_path, dimensions=[4, 6, 8])
    cli.main([cfg, "--output-dir", str(tmp_path)])
    assert [config.d for config, _ in recorded_runs] == [4, 6, 8]
    output = capsys.readouterr().out
    assert "Dimension d=4 (1/3)" in output
    assert "Dimension d=8 (3/3)" in output


def test_main_defaults_to_ideal_device(recorded_runs, tmp_path):
    cfg = _write_config(tmp_path, dimensions=[4])
    cli.main([cfg, "--output-dir", str(tmp_path)])
    _, device = recorded_runs[0]
    assert device.name == "ideal"
    assert device.t1_us == [] and device.t2_us == []


def test_main_loads_named_device(recorded_runs, tmp_path):
    cfg = _write_config(tmp_path, dimensions=[4])
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"name": "sweep_s", "t1_us": [5], "t2_us": [10]}))
    cli.main([cfg, "--device", str(dev), "--output-dir", str(tmp_path)])
    _, device = recorded_runs[0]
    assert device.name == "sweep_s"
    assert device.t1_us == [5]


def test_main_rejects_unknown_benchmark(recorded_runs, tmp_path):
    cfg = _write_config(tmp_path, benchmark_name="nope")
    with pytest.raises(Exception):
        cli.main([cfg, "--output-dir", str(tmp_path)])
    assert recorded_runs == []


def test_main_applies_output_dir(monkeypatch, recorded_runs, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "set_output_dir", lambda path: captured.setdefault("path", path))
    cfg = _write_config(tmp_path, dimensions=[4])
    cli.main([cfg, "--output-dir", str(tmp_path)])
    assert captured["path"] == str(tmp_path)
