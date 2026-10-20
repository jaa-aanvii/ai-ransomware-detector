"""
response.py

The "automatically takes preventive action" half of the project, which
was missing before -- everything up to now only detects and prints.

Design choices, and why:

  * DRY-RUN BY DEFAULT. Suspending or killing a PID based on an
    unvalidated model is genuinely risky (a false positive could kill a
    legitimate backup job or IDE mid-write and corrupt the very files
    you're trying to protect). ResponseEngine only LOGS what it *would*
    do unless you explicitly pass dry_run=False, so accidentally
    running this against your own machine can't do damage.

  * SUSPEND BEFORE KILL. psutil's suspend() pauses a process without
    losing its state (like SIGSTOP) -- reversible, unlike terminate()/
    kill(). The engine suspends first and only escalates to termination
    if the same PID keeps triggering alerts, giving a human a window to
    intervene.

  * EVERY ACTION IS LOGGED. Incidents are appended as JSON lines to an
    audit log so you have an evidence trail of what the system saw and
    did -- useful both for debugging false positives and for a project
    writeup/demo.

  * FILE QUARANTINE IS BEST-EFFORT AND NON-DESTRUCTIVE. It copies
    recently-touched files to a quarantine folder; it never deletes or
    modifies the originals. Ransomware recovery tooling should always
    prefer "make a safe copy" over "touch the evidence".
"""

import json
import os
import time

import psutil


class ResponseEngine:
    def __init__(
        self,
        threshold=0.85,
        dry_run=True,
        log_path="logs/incidents.jsonl",
        kill_after_repeat_alerts=3,
        quarantine_dir=None,
    ):
        self.threshold = threshold
        self.dry_run = dry_run
        self.log_path = log_path
        self.kill_after_repeat_alerts = kill_after_repeat_alerts
        self.quarantine_dir = quarantine_dir

        self._alert_counts = {}  # pid -> consecutive alert count

        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        if self.quarantine_dir:
            os.makedirs(self.quarantine_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------

    def _log(self, record: dict):
        record["timestamp"] = time.time()
        record["dry_run"] = self.dry_run

        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------

    def _process_name(self, pid):
        try:
            return psutil.Process(pid).name()
        except psutil.NoSuchProcess:
            return "<unknown>"

    def _suspend(self, pid):
        if self.dry_run:
            return "would_suspend"

        try:
            psutil.Process(pid).suspend()
            return "suspended"
        except psutil.NoSuchProcess:
            return "process_gone"
        except psutil.AccessDenied:
            return "access_denied"

    def _terminate(self, pid):
        if self.dry_run:
            return "would_terminate"

        try:
            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                proc.kill()
            return "terminated"
        except psutil.NoSuchProcess:
            return "process_gone"
        except psutil.AccessDenied:
            return "access_denied"

    def _quarantine_hint(self, pid):
        """Best-effort: copy currently-open files of the offending
        process into the quarantine directory as read-only snapshots.
        Skipped entirely if quarantine_dir wasn't configured."""
        if not self.quarantine_dir:
            return None

        copied = []
        try:
            proc = psutil.Process(pid)
            for f in proc.open_files():
                try:
                    dest = os.path.join(
                        self.quarantine_dir,
                        f"{pid}_{os.path.basename(f.path)}",
                    )
                    if not self.dry_run:
                        with open(f.path, "rb") as src, open(dest, "wb") as out:
                            out.write(src.read())
                    copied.append(f.path)
                except (OSError, PermissionError):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return copied

    # ------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------

    def handle_prediction(self, pid, prediction, sequence=None):
        """Call this once per model prediction. Only acts when the
        RANSOMWARE probability clears `threshold`; otherwise resets the
        repeat-alert counter for that PID (so a single fluke reading
        doesn't count toward escalation)."""

        probability = prediction["probability"]
        label = prediction["label"]

        if label != "RANSOMWARE" or probability < self.threshold:
            self._alert_counts.pop(pid, None)
            return {"action": "none", "reason": "below_threshold"}

        self._alert_counts[pid] = self._alert_counts.get(pid, 0) + 1
        repeat_count = self._alert_counts[pid]

        process_name = self._process_name(pid)
        quarantined = self._quarantine_hint(pid)

        if repeat_count < self.kill_after_repeat_alerts:
            action_result = self._suspend(pid)
            action = "suspend"
        else:
            action_result = self._terminate(pid)
            action = "terminate"

        record = {
            "pid": pid,
            "process_name": process_name,
            "probability": probability,
            "repeat_count": repeat_count,
            "action": action,
            "action_result": action_result,
            "quarantined_files": quarantined,
        }

        self._log(record)

        return record
