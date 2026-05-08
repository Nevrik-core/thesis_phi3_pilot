import os
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import psutil


MB = 1024 * 1024


def _safe_rss_mb(process: psutil.Process) -> float | None:
    try:
        return process.memory_info().rss / MB
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def _safe_name_and_cmdline(process: psutil.Process) -> str:
    try:
        name = process.name() or ""
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        name = ""

    try:
        cmdline = " ".join(process.cmdline() or [])
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        cmdline = ""

    return f"{name} {cmdline}".lower()


def find_processes_by_terms(terms: Iterable[str]) -> list[psutil.Process]:
    lowered_terms = [t.lower() for t in terms]
    current_pid = os.getpid()
    matches = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid == current_pid:
                continue

            haystack = _safe_name_and_cmdline(proc)

            if any(term in haystack for term in lowered_terms):
                matches.append(proc)

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return matches


def current_process_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / MB


def system_used_memory_mb() -> float:
    return psutil.virtual_memory().used / MB


def total_rss_mb(processes: list[psutil.Process]) -> float:
    values = [_safe_rss_mb(p) for p in processes]
    return sum(v for v in values if v is not None)


@dataclass
class ResourceMetrics:
    current_process_rss_before_mb: float
    current_process_rss_after_mb: float
    current_process_peak_rss_mb: float

    external_process_rss_before_mb: float | None
    external_process_rss_after_mb: float | None
    external_process_peak_rss_mb: float | None
    external_process_count_before: int | None
    external_process_count_after: int | None

    system_used_memory_before_mb: float
    system_used_memory_after_mb: float
    system_used_memory_peak_mb: float

    poll_interval_sec: float

    def to_dict(
        self,
        current_label: str = "current_process",
        external_label: str = "external_process",
    ) -> dict:
        return {
            f"{current_label}_rss_before_mb": round(self.current_process_rss_before_mb, 2),
            f"{current_label}_rss_after_mb": round(self.current_process_rss_after_mb, 2),
            f"{current_label}_peak_rss_mb": round(self.current_process_peak_rss_mb, 2),

            f"{external_label}_rss_before_mb": (
                round(self.external_process_rss_before_mb, 2)
                if self.external_process_rss_before_mb is not None
                else None
            ),
            f"{external_label}_rss_after_mb": (
                round(self.external_process_rss_after_mb, 2)
                if self.external_process_rss_after_mb is not None
                else None
            ),
            f"{external_label}_peak_rss_mb": (
                round(self.external_process_peak_rss_mb, 2)
                if self.external_process_peak_rss_mb is not None
                else None
            ),
            f"{external_label}_count_before": self.external_process_count_before,
            f"{external_label}_count_after": self.external_process_count_after,

            "system_used_memory_before_mb": round(self.system_used_memory_before_mb, 2),
            "system_used_memory_after_mb": round(self.system_used_memory_after_mb, 2),
            "system_used_memory_peak_mb": round(self.system_used_memory_peak_mb, 2),

            "resource_poll_interval_sec": self.poll_interval_sec,
        }


class ResourceMonitor:
    def __init__(
        self,
        external_process_terms: tuple[str, ...] | None = None,
        poll_interval_sec: float = 0.25,
    ):
        self.external_process_terms = external_process_terms
        self.poll_interval_sec = poll_interval_sec

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._current_before = 0.0
        self._current_after = 0.0
        self._current_peak = 0.0

        self._external_before: float | None = None
        self._external_after: float | None = None
        self._external_peak: float | None = None
        self._external_count_before: int | None = None
        self._external_count_after: int | None = None

        self._system_before = 0.0
        self._system_after = 0.0
        self._system_peak = 0.0

    def __enter__(self):
        self._current_before = current_process_rss_mb()
        self._current_peak = self._current_before

        self._system_before = system_used_memory_mb()
        self._system_peak = self._system_before

        if self.external_process_terms:
            processes = find_processes_by_terms(self.external_process_terms)
            self._external_count_before = len(processes)
            self._external_before = total_rss_mb(processes)
            self._external_peak = self._external_before

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        self._sample()

        self._current_after = current_process_rss_mb()
        self._system_after = system_used_memory_mb()

        if self.external_process_terms:
            processes = find_processes_by_terms(self.external_process_terms)
            self._external_count_after = len(processes)
            self._external_after = total_rss_mb(processes)

        return False

    def _poll_loop(self):
        while not self._stop_event.is_set():
            self._sample()
            time.sleep(self.poll_interval_sec)

    def _sample(self):
        current_rss = current_process_rss_mb()
        self._current_peak = max(self._current_peak, current_rss)

        system_used = system_used_memory_mb()
        self._system_peak = max(self._system_peak, system_used)

        if self.external_process_terms:
            processes = find_processes_by_terms(self.external_process_terms)
            external_rss = total_rss_mb(processes)

            if self._external_peak is None:
                self._external_peak = external_rss
            else:
                self._external_peak = max(self._external_peak, external_rss)

    def metrics(self) -> ResourceMetrics:
        return ResourceMetrics(
            current_process_rss_before_mb=self._current_before,
            current_process_rss_after_mb=self._current_after,
            current_process_peak_rss_mb=self._current_peak,

            external_process_rss_before_mb=self._external_before,
            external_process_rss_after_mb=self._external_after,
            external_process_peak_rss_mb=self._external_peak,
            external_process_count_before=self._external_count_before,
            external_process_count_after=self._external_count_after,

            system_used_memory_before_mb=self._system_before,
            system_used_memory_after_mb=self._system_after,
            system_used_memory_peak_mb=self._system_peak,

            poll_interval_sec=self.poll_interval_sec,
        )