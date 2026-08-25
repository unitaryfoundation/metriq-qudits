from metriq_qudits.benchmarks.benchmark import Benchmark, BenchmarkResult
from metriq_qudits.benchmarks.quantum_volume import QVResult, QuantumVolume
from metriq_qudits.constants import JobType

BENCHMARK_HANDLERS: dict[JobType, type[Benchmark]] = {
    JobType.QUANTUM_VOLUME: QuantumVolume,
}

BENCHMARK_RESULT_CLASSES: dict[JobType, type[BenchmarkResult]] = {
    JobType.QUANTUM_VOLUME: QVResult,
}
