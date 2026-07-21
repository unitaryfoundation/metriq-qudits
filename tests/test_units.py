"""Unit tests for metriq_qudits.physics.units."""

import math

import pytest

from metriq_qudits.physics.units import (
    NS_PER_US,
    angular_frequency_from_hz,
    angular_frequency_from_khz,
    angular_frequency_from_mhz,
    inverse_time_from_khz,
    inverse_time_from_us,
)


def test_angular_frequency_from_hz_known_value():
    # 1 Hz -> 2π rad/s -> 2π·1e-9 rad/ns
    assert angular_frequency_from_hz(1.0) == pytest.approx(2 * math.pi * 1e-9)


def test_angular_frequency_khz_and_mhz_scale_from_hz():
    assert angular_frequency_from_khz(3.3) == pytest.approx(angular_frequency_from_hz(3.3e3))
    assert angular_frequency_from_mhz(200.0) == pytest.approx(angular_frequency_from_hz(200.0e6))


def test_angular_frequency_is_linear_and_zero_at_zero():
    assert angular_frequency_from_hz(0.0) == 0.0
    assert angular_frequency_from_khz(-32.8) == pytest.approx(-angular_frequency_from_khz(32.8))


def test_inverse_time_from_khz_has_no_two_pi():
    # decay rate: 1 kHz -> 1e-6 /ns, with NO factor of 2π
    assert inverse_time_from_khz(1.0) == pytest.approx(1e-6)
    assert inverse_time_from_khz(1.0) == pytest.approx(
        angular_frequency_from_khz(1.0) / (2 * math.pi)
    )


def test_inverse_time_from_us_known_value():
    # 1 µs lifetime -> 1/(1000 ns) = 1e-3 /ns
    assert inverse_time_from_us(1.0) == pytest.approx(1.0 / NS_PER_US)
    assert inverse_time_from_us(2.0) == pytest.approx(5e-4)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_inverse_time_from_us_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        inverse_time_from_us(bad)


def test_ns_per_us_constant():
    assert NS_PER_US == 1_000.0
