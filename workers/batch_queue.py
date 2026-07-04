"""Batch transcription queue — sequential multi-file processing."""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class BatchJob:
    """A single batch transcription job."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"

    def __init__(self, file_path, backend=None, model=None, language=None):
        self.file_path = file_path
        self.backend = backend
        self.model = model
        self.language = language
        self.status = self.QUEUED
        self.progress = 0.0
        self.result_segments = []
        self.result_text = ""
        self.error_message = ""
        self.created_at = time.time()
        self.elapsed = 0.0


class BatchQueue:
    """Sequential batch processor for multiple audio files.

    Usage:
        queue = BatchQueue(on_job_done=callback, on_job_error=callback)
        queue.add("file1.wav", backend="crispasr:parakeet", model="auto")
        queue.add("file2.wav", backend="crispasr:parakeet", model="auto")
        queue.start()
    """

    def __init__(
        self, on_job_started=None, on_job_progress=None, on_job_done=None, on_job_error=None
    ):
        self._jobs = []
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._on_job_started = on_job_started
        self._on_job_progress = on_job_progress
        self._on_job_done = on_job_done
        self._on_job_error = on_job_error

    @property
    def jobs(self):
        with self._lock:
            return list(self._jobs)

    @property
    def is_running(self):
        return self._running

    def add(self, file_path, backend=None, model=None, language=None):
        """Add a file to the queue."""
        job = BatchJob(file_path, backend=backend, model=model, language=language)
        with self._lock:
            self._jobs.append(job)
        return job

    def remove(self, index):
        """Remove a queued job by index."""
        with self._lock:
            if 0 <= index < len(self._jobs) and self._jobs[index].status == BatchJob.QUEUED:
                self._jobs.pop(index)

    def clear_completed(self):
        """Remove all completed and errored jobs."""
        with self._lock:
            self._jobs = [j for j in self._jobs if j.status in (BatchJob.QUEUED, BatchJob.RUNNING)]

    def start(self):
        """Start processing the queue in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the queue to stop after the current job."""
        self._running = False

    def _drain(self):
        """Process jobs sequentially."""
        while self._running:
            job = self._next_queued()
            if job is None:
                break
            self._process_job(job)
        self._running = False

    def _next_queued(self):
        with self._lock:
            for job in self._jobs:
                if job.status == BatchJob.QUEUED:
                    return job
        return None

    def _process_job(self, job):
        """Process a single job."""
        job.status = BatchJob.RUNNING
        if self._on_job_started:
            self._on_job_started(job)

        t0 = time.time()
        try:
            from workers.transcription.backends import get_backend

            backend_name = job.backend or "crispasr"
            model = job.model or "auto"

            backend_obj = get_backend(
                backend_name, model_id=model, device="cpu", language=job.language
            )
            try:
                audio_path = backend_obj.preprocess_audio(job.file_path)
                segments = []
                for start, end, text in backend_obj.transcribe(audio_path):
                    segments.append((start, end, text))
                    job.result_segments = segments
                    if self._on_job_progress:
                        self._on_job_progress(job, len(segments))
                job.result_segments = segments
                job.result_text = "\n".join(text for _, _, text in segments)
                job.status = BatchJob.DONE
            finally:
                backend_obj.cleanup()
        except Exception as e:
            job.status = BatchJob.ERROR
            job.error_message = str(e)
            logger.error("Batch job failed for %s: %s", job.file_path, e)
            if self._on_job_error:
                self._on_job_error(job)
        finally:
            job.elapsed = time.time() - t0

        if job.status == BatchJob.DONE and self._on_job_done:
            self._on_job_done(job)

    @property
    def summary(self):
        """Return counts by status."""
        counts = {BatchJob.QUEUED: 0, BatchJob.RUNNING: 0, BatchJob.DONE: 0, BatchJob.ERROR: 0}
        with self._lock:
            for job in self._jobs:
                counts[job.status] = counts.get(job.status, 0) + 1
        return counts
