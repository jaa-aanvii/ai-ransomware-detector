from collections import defaultdict


class PIDTracker:

    def __init__(self):

        self.processes = defaultdict(
            lambda: {
                "last_timestamp": None,
                "windows": 0
            }
        )

    def update(
        self,
        pid,
        timestamp
    ):

        process = self.processes[pid]

        process["last_timestamp"] = timestamp

        process["windows"] += 1

    def get(self, pid):

        return self.processes.get(
            pid
        )

    def clear(self, pid):

        if pid in self.processes:
            del self.processes[pid]