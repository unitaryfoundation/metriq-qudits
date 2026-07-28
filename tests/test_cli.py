"""Tests for the metriq_qudits.cli argument parsing and orchestration.

``main`` is exercised with ``run_experiment`` monkeypatched to a recorder so the
expensive compile/pulse/simulate pipeline never runs; only the CLI's own
argument handling (config selection, T1/T2 resolution, flag mapping) is tested.
"""

import numpy as np
import pytest

from metriq_qudits import cli
from metriq_qudits.simulation.sweep import T1_VALUES, T2_VALUES


@pytest.fixture
def recorded_runs(monkeypatch):
    """Replace run_experiment with a recorder and return the captured calls."""
    calls = []

    def recorder(config, **kwargs):
        calls.append((config, kwargs))

    monkeypatch.setattr(cli, "run_experiment", recorder)
    return calls


# ---------------------------------------------------------------------------
# _parse_args: pure argument parsing
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    _, args = cli._parse_args([])
    assert args.configs is None
    assert args.t1 is None
    assert args.t2 is None
    assert args.skip_sweep is False
    assert args.overwrite is False
    assert args.no_phase_correction is False
    assert args.plot is False
    assert args.plot_only is False
    assert args.backend == "dynamiqs"
    assert args.output_dir is None


def test_parse_args_t1_t2_are_float_lists():
    _, args = cli._parse_args(["--t1", "5", "10", "20", "--t2", "10", "40"])
    assert args.t1 == [5.0, 10.0, 20.0]
    assert args.t2 == [10.0, 40.0]


def test_parse_args_rejects_unknown_backend():
    with pytest.raises(SystemExit):
        cli._parse_args(["--backend", "cirq"])


def test_parse_args_n_jobs_defaults_to_env_value():
    _, args = cli._parse_args([])
    assert args.n_jobs == cli.N_JOBS


def test_parse_args_n_jobs_flag_overrides_default():
    _, args = cli._parse_args(["--n-jobs", "4"])
    assert args.n_jobs == 4


def test_parse_args_rejects_non_integer_n_jobs():
    with pytest.raises(SystemExit):
        cli._parse_args(["--n-jobs", "eight"])


def test_parse_args_store_true_flags():
    _, args = cli._parse_args(
        ["--skip-sweep", "--overwrite", "--no-phase-correction", "--plot"]
    )
    assert args.skip_sweep is True
    assert args.overwrite is True
    assert args.no_phase_correction is True
    assert args.plot is True


# ---------------------------------------------------------------------------
# main: config selection
# ---------------------------------------------------------------------------

def test_main_default_runs_all_configs(recorded_runs):
    cli.main(["--skip-sweep"])
    run_keys = [config.key for config, _ in recorded_runs]
    assert run_keys == [config.key for config in cli.CONFIGS]


def test_main_selects_requested_configs_in_order(recorded_runs):
    cli.main(["--configs", "d6", "d4", "--skip-sweep"])
    run_keys = [config.key for config, _ in recorded_runs]
    assert run_keys == ["d6", "d4"]


def test_main_rejects_unknown_config(recorded_runs):
    with pytest.raises(SystemExit):
        cli.main(["--configs", "d99"])
    assert recorded_runs == []


# ---------------------------------------------------------------------------
# main: T1/T2 grid resolution and validation
# ---------------------------------------------------------------------------

def test_main_uses_default_grids_when_unset(recorded_runs):
    cli.main(["--configs", "d4", "--skip-sweep"])
    _, kwargs = recorded_runs[0]
    assert np.array_equal(kwargs["t1_values"], T1_VALUES)
    assert np.array_equal(kwargs["t2_values"], T2_VALUES)


def test_main_converts_microseconds_to_seconds(recorded_runs):
    cli.main(["--configs", "d4", "--skip-sweep", "--t1", "5", "10", "--t2", "20"])
    _, kwargs = recorded_runs[0]
    assert np.allclose(kwargs["t1_values"], np.array([5.0, 10.0]) * 1e-6)
    assert np.allclose(kwargs["t2_values"], np.array([20.0]) * 1e-6)


@pytest.mark.parametrize("flag", ["--t1", "--t2"])
def test_main_rejects_nonpositive_lifetimes(recorded_runs, flag):
    with pytest.raises(SystemExit):
        cli.main(["--configs", "d4", "--skip-sweep", flag, "0"])
    assert recorded_runs == []


# ---------------------------------------------------------------------------
# main: flag-to-keyword mapping
# ---------------------------------------------------------------------------

def test_main_flag_mapping(recorded_runs):
    cli.main(
        ["--configs", "d4", "--skip-sweep", "--no-phase-correction", "--overwrite",
         "--backend", "qutip"]
    )
    _, kwargs = recorded_runs[0]
    assert kwargs["include_noise_sweep"] is False
    assert kwargs["correct_phases"] is False
    assert kwargs["overwrite"] is True
    assert kwargs["backend"] == "qutip"


def test_main_includes_sweep_by_default(recorded_runs):
    cli.main(["--configs", "d4"])
    _, kwargs = recorded_runs[0]
    assert kwargs["include_noise_sweep"] is True
    assert kwargs["correct_phases"] is True


def test_main_n_jobs_flag_passed_to_run_experiment(recorded_runs):
    cli.main(["--configs", "d4", "--skip-sweep", "--n-jobs", "4"])
    _, kwargs = recorded_runs[0]
    assert kwargs["n_jobs"] == 4


def test_main_n_jobs_defaults_to_module_value(recorded_runs):
    cli.main(["--configs", "d4", "--skip-sweep"])
    _, kwargs = recorded_runs[0]
    assert kwargs["n_jobs"] == cli.N_JOBS


def test_main_output_dir_is_applied(monkeypatch, recorded_runs, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "set_output_dir", lambda path: captured.setdefault("path", path))
    cli.main(["--configs", "d4", "--skip-sweep", "--output-dir", str(tmp_path)])
    assert captured["path"] == str(tmp_path)


# ---------------------------------------------------------------------------
# main: --plot-only regenerates figures without running the pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def recorded_plots(monkeypatch):
    """Replace plot_results.main with a recorder counting invocations."""
    import metriq_qudits.plot_results as plot_results

    calls = []
    monkeypatch.setattr(plot_results, "main", lambda: calls.append(True))
    return calls


def test_plot_only_plots_and_skips_pipeline(recorded_runs, recorded_plots):
    cli.main(["--plot-only"])
    assert recorded_plots == [True]
    assert recorded_runs == []


def test_plot_only_ignores_configs_and_grids(recorded_runs, recorded_plots):
    # --plot-only short-circuits before config validation, so a bogus config
    # and custom grids are simply ignored rather than raising.
    cli.main(["--plot-only", "--configs", "d99", "--t1", "-5"])
    assert recorded_plots == [True]
    assert recorded_runs == []


def test_plot_only_applies_output_dir_before_plotting(
    monkeypatch, recorded_runs, recorded_plots, tmp_path
):
    captured = {}
    monkeypatch.setattr(cli, "set_output_dir", lambda path: captured.setdefault("path", path))
    cli.main(["--plot-only", "--output-dir", str(tmp_path)])
    assert captured["path"] == str(tmp_path)
    assert recorded_plots == [True]
