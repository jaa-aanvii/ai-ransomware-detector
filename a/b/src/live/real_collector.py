"""
real_collector.py

Live replacement for DatasetEventCollector. Watches a real directory on
disk and produces the SAME 8-feature window rows the model was trained
on (write_rate, rename_rate, unique_dirs, files_modified, entropy,
dir_dispersion, proc_cpu_pct, handle_count), computed from actual OS
activity instead of replayed from an Excel file.

IMPORTANT / HONEST LIMITATION
------------------------------
Standard file-system watchers (watchdog/inotify) tell you WHAT happened
to a file, not WHICH PROCESS did it. True per-PID file attribution needs
kernel-level hooks (Linux fanotify with FAN_REPORT_PID + FAN_REPORT_FID,
or an eBPF program, or on Windows a minifilter/ETW provider) running as
root/admin -- that is a separate, OS-specific piece of work beyond a
portable Python script.

This collector uses a documented, best-effort heuristic instead:
  1. watchdog reports file-system events for the watched directory
     system-wide (created / modified / moved / deleted), each window.
  2. psutil polls every running process's IO counters each window.
  3. The window's file-system activity is attributed to whichever
     process had the largest WRITE-BYTE delta during that window --
     i.e. "whoever wrote the most bytes is probably who did this".

This is good enough to demo and test the model against real behavior
on a single dev machine (e.g. running a benign script vs. a
self-authored encryption-simulation script in a scratch directory).
It is NOT a production-grade attribution mechanism and should not be
described as one. Swap this module out for a fanotify/ETW-based
collector before relying on this for real endpoint protection.
"""

import math
import os
import time
import threading
from collections import defaultdict, deque

import psutil
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


FEATURE_COLUMNS = [
    "write_rate",
    "rename_rate",
    "unique_dirs",
    "files_modified",
    "entropy",
    "dir_dispersion",
    "proc_cpu_pct",
    "handle_count",
]

# How many bytes to sample from a modified file when estimating entropy.
# Sampling (not reading whole files) keeps this cheap enough to run on
# every write event without noticeably slowing the monitored system.
ENTROPY_SAMPLE_BYTES = 4096


def shannon_entropy(data: bytes) -> float:
    """Shannon entropy in bits/byte (0-8), same scale as the synthetic
    telemetry generator used for training."""
    if not data:
        return 0.0

    counts = defaultdict(int)
    for b in data:
        counts[b] += 1

    length = len(data)
    entropy = 0.0

    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


class _WatchHandler(FileSystemEventHandler):
    """Buffers raw watchdog events; the collector drains this on each tick."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.events = []  # list of (kind, src_path, dest_path_or_None, timestamp)

    def _push(self, kind, src_path, dest_path=None):
        with self.lock:
            self.events.append((kind, src_path, dest_path, time.time()))

    def on_created(self, event):
        if not event.is_directory:
            self._push("created", event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._push("modified", event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._push("moved", event.src_path, event.dest_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._push("deleted", event.src_path)

    def drain(self):
        with self.lock:
            events, self.events = self.events, []
        return events


class RealTimeEventCollector:
    """
    Drop-in replacement for DatasetEventCollector. Instead of
    `.events()` reading rows from a spreadsheet, it watches `watch_path`
    on disk and yields one feature row per (attributed PID, window).

    Usage matches the dataset collector:
        for event in collector.events():
            # event["pid"], event["timestamp"], event["features"], event["label"]
    """

    def __init__(
        self,
        watch_path,
        window_size_ms=500,
        recursive=True,
    ):
        self.watch_path = os.path.abspath(watch_path)
        self.window_size_s = window_size_ms / 1000.0
        self.recursive = recursive

        if not os.path.isdir(self.watch_path):
            raise ValueError(
                f"watch_path does not exist or is not a directory: {self.watch_path}"
            )

        self._handler = _WatchHandler()
        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            self.watch_path,
            recursive=self.recursive,
        )

        # Rolling IO-counter snapshots per PID, used to compute per-window
        # write-byte deltas for process attribution and CPU deltas.
        self._last_io = {}
        self._start_time = time.time()

        # Cached rough estimate of "how many directories could this app
        # plausibly touch" -- used to normalize dir_dispersion into 0-1,
        # matching the synthetic feature's scale. Computed once at start.
        self._max_dirs = max(1, self._count_dirs(self.watch_path))

    @staticmethod
    def _count_dirs(root):
        count = 0
        for _, dirnames, _ in os.walk(root):
            count += len(dirnames)
        return count

    def _snapshot_process_io(self):
        """Returns {pid: (write_bytes, cpu_percent, num_handles)} for
        every process we can read right now. Best-effort: processes
        that disappear or deny access mid-scan are skipped."""
        snapshot = {}

        for proc in psutil.process_iter(
            ["pid", "name"]
        ):
            try:
                pid = proc.info["pid"]
                io = proc.io_counters()
                cpu = proc.cpu_percent(interval=None)

                if hasattr(proc, "num_fds"):
                    handles = proc.num_fds()  # POSIX
                elif hasattr(proc, "num_handles"):
                    handles = proc.num_handles()  # Windows
                else:
                    handles = 0

                snapshot[pid] = {
                    "write_bytes": io.write_bytes,
                    "cpu_percent": cpu,
                    "handle_count": handles,
                }

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
            ):
                continue

        return snapshot

    def _attribute_pid(self, prev_io, curr_io):
        """Returns (pid, cpu_percent, handle_count) for the process with
        the largest write-byte delta since the last snapshot, or None if
        nothing wrote anything this window."""
        best_pid = None
        best_delta = 0

        for pid, curr in curr_io.items():
            prev = prev_io.get(pid)

            if prev is None:
                continue

            delta = curr["write_bytes"] - prev["write_bytes"]

            if delta > best_delta:
                best_delta = delta
                best_pid = pid

        if best_pid is None:
            return None

        return (
            best_pid,
            curr_io[best_pid]["cpu_percent"],
            curr_io[best_pid]["handle_count"],
        )

    def events(self):
        """Generator that blocks in real time, yielding one row per
        window (only when there was file-system activity to report)."""
        self._observer.start()
        self._last_io = self._snapshot_process_io()

        try:
            while True:
                time.sleep(self.window_size_s)

                raw_events = self._handler.drain()
                curr_io = self._snapshot_process_io()

                if not raw_events:
                    self._last_io = curr_io
                    continue

                write_count = sum(
                    1 for kind, *_ in raw_events if kind in ("created", "modified")
                )
                rename_count = sum(
                    1 for kind, *_ in raw_events if kind == "moved"
                )
                touched_paths = {
                    e[1] for e in raw_events
                } | {
                    e[2] for e in raw_events if e[2]
                }
                touched_dirs = {
                    os.path.dirname(p) for p in touched_paths
                }

                # Sample entropy from modified/created files only.
                entropy_samples = []
                for kind, src_path, _, _ in raw_events:
                    if kind in ("created", "modified") and os.path.isfile(src_path):
                        try:
                            with open(src_path, "rb") as fh:
                                data = fh.read(ENTROPY_SAMPLE_BYTES)
                            entropy_samples.append(shannon_entropy(data))
                        except OSError:
                            continue

                mean_entropy = (
                    sum(entropy_samples) / len(entropy_samples)
                    if entropy_samples
                    else 0.0
                )

                dir_dispersion = min(
                    1.0, len(touched_dirs) / self._max_dirs
                )

                attribution = self._attribute_pid(self._last_io, curr_io)

                timestamp = time.time() - self._start_time

                if attribution is None:
                    # File activity happened but we couldn't attribute it
                    # to a specific PID (e.g. it wrote via a buffered
                    # handle we didn't see delta on). Report under a
                    # sentinel PID rather than silently dropping data.
                    pid, cpu_pct, handle_count = -1, 0.0, 0
                else:
                    pid, cpu_pct, handle_count = attribution

                yield {
                    "pid": pid,
                    "timestamp": timestamp,
                    "features": [
                        float(write_count),
                        float(rename_count),
                        float(len(touched_dirs)),
                        float(len(touched_paths)),
                        float(mean_entropy),
                        float(dir_dispersion),
                        float(cpu_pct),
                        float(handle_count),
                    ],
                    # No ground truth in live mode -- this is what we're
                    # trying to find out, not something we already know.
                    "label": None,
                }

                self._last_io = curr_io

        finally:
            self._observer.stop()
            self._observer.join(timeout=2)

    def stop(self):
        self._observer.stop()
