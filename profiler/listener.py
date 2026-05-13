# listener.py

import time
import threading


class QueryProfilerListener:
    """
    Collects stage metrics using Spark's StatusTracker API.

    Key findings from API inspection:
    - StatusTracker methods: getActiveJobsIds, getActiveStageIds,
      getJobIdsForGroup(group_str), getJobInfo(jid), getStageInfo(sid)
    - SparkJobInfo fields: jobId, status, stageIds (JavaArray — use list())
    - SparkStageInfo fields: stageId, currentAttemptId, numTasks,
      numActiveTasks, numCompletedTasks, numFailedTasks, name
      (NO submissionTime / completionTime — not exposed in Python API)
    - getJobIdsForGroup(None) returns ALL jobs; getJobIdsForGroup("") returns none
    - Completed jobs are still visible via getJobInfo after execution ends
    """

    def __init__(self, sc):
        self._sc              = sc
        self._tracker         = sc.statusTracker()
        self.records          = []
        self._stop_event      = threading.Event()
        self._poll_thread     = None
        self._seen_stage_ids  = set()
        self._job_id_baseline = self._get_all_job_ids()  # snapshot before query

    def _get_all_job_ids(self):
        """Return the set of all currently known job IDs."""
        try:
            return set(self._tracker.getJobIdsForGroup(None))
        except Exception:
            return set()

    def start(self):
        """Begin background polling — captures stages during execution."""
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True
        )
        self._poll_thread.start()

    def stop(self):
        """
        Stop polling, then do a final scan of all jobs created since
        start() was called to catch any stages missed during polling.
        """
        self._stop_event.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        # Final flush: scan every new job ID created during the query
        self._flush_new_jobs()

    def _poll_loop(self):
        while not self._stop_event.is_set():
            self._flush_active_stages()
            time.sleep(0.3)

    def _flush_active_stages(self):
        """Poll currently active stage IDs (fast path during execution)."""
        try:
            active_ids = list(self._tracker.getActiveStageIds())
        except Exception:
            return
        for sid in active_ids:
            self._record_stage(sid)

    def _flush_new_jobs(self):
        """
        After execution completes, scan all job IDs that didn't exist
        before the query ran and record their stages. This is the
        reliable catch-all since StatusTracker retains completed job info.
        """
        try:
            current_ids = set(self._tracker.getJobIdsForGroup(None))
        except Exception:
            return

        new_job_ids = current_ids - self._job_id_baseline

        for jid in new_job_ids:
            try:
                job_info = self._tracker.getJobInfo(jid)
                if job_info is None:
                    continue
                for sid in list(job_info.stageIds):
                    self._record_stage(sid, job_id=jid,
                                       job_status=job_info.status)
            except Exception:
                continue

    def _record_stage(self, sid, job_id=None, job_status=None):
        """Record a stage if not already seen."""
        if sid in self._seen_stage_ids:
            return
        try:
            info = self._tracker.getStageInfo(sid)
            if info is None:
                return

            self._seen_stage_ids.add(sid)
            self.records.append({
                "ts":                   time.time(),
                "record_type":          "stage",
                "stage_id":             sid,
                "stage_name":           info.name,
                "current_attempt_id":   info.currentAttemptId,
                "num_tasks":            info.numTasks,
                "num_completed_tasks":  info.numCompletedTasks,
                "num_failed_tasks":     info.numFailedTasks,
                "num_active_tasks":     info.numActiveTasks,
                "job_id":               job_id,
                "job_status":           job_status,
            })
        except Exception:
            return