"""Test batch queue logic."""

import unittest


class TestBatchJob(unittest.TestCase):
    def test_initial_state(self):
        from workers.batch_queue import BatchJob

        job = BatchJob("/tmp/audio.wav", backend="whisper", model="auto")
        self.assertEqual(job.file_path, "/tmp/audio.wav")
        self.assertEqual(job.status, BatchJob.QUEUED)
        self.assertEqual(job.progress, 0.0)
        self.assertEqual(job.result_segments, [])

    def test_status_constants(self):
        from workers.batch_queue import BatchJob

        self.assertEqual(BatchJob.QUEUED, "queued")
        self.assertEqual(BatchJob.RUNNING, "running")
        self.assertEqual(BatchJob.DONE, "done")
        self.assertEqual(BatchJob.ERROR, "error")


class TestBatchQueue(unittest.TestCase):
    def test_add_jobs(self):
        from workers.batch_queue import BatchQueue

        q = BatchQueue()
        q.add("file1.wav", backend="whisper")
        q.add("file2.wav", backend="whisper")
        self.assertEqual(len(q.jobs), 2)

    def test_remove_queued(self):
        from workers.batch_queue import BatchQueue

        q = BatchQueue()
        q.add("file1.wav")
        q.add("file2.wav")
        q.remove(0)
        self.assertEqual(len(q.jobs), 1)
        self.assertEqual(q.jobs[0].file_path, "file2.wav")

    def test_remove_out_of_range(self):
        from workers.batch_queue import BatchQueue

        q = BatchQueue()
        q.add("file1.wav")
        q.remove(5)  # Should not crash
        self.assertEqual(len(q.jobs), 1)

    def test_clear_completed(self):
        from workers.batch_queue import BatchJob, BatchQueue

        q = BatchQueue()
        j1 = q.add("file1.wav")
        q.add("file2.wav")
        j3 = q.add("file3.wav")
        j1.status = BatchJob.DONE
        j3.status = BatchJob.ERROR
        q.clear_completed()
        self.assertEqual(len(q.jobs), 1)
        self.assertEqual(q.jobs[0].file_path, "file2.wav")

    def test_summary(self):
        from workers.batch_queue import BatchJob, BatchQueue

        q = BatchQueue()
        j1 = q.add("f1.wav")
        j2 = q.add("f2.wav")
        q.add("f3.wav")
        j1.status = BatchJob.DONE
        j2.status = BatchJob.RUNNING
        s = q.summary
        self.assertEqual(s[BatchJob.DONE], 1)
        self.assertEqual(s[BatchJob.RUNNING], 1)
        self.assertEqual(s[BatchJob.QUEUED], 1)

    def test_not_running_initially(self):
        from workers.batch_queue import BatchQueue

        q = BatchQueue()
        self.assertFalse(q.is_running)

    def test_stop(self):
        from workers.batch_queue import BatchQueue

        q = BatchQueue()
        q._running = True
        q.stop()
        self.assertFalse(q._running)

    def test_next_queued_empty(self):
        from workers.batch_queue import BatchQueue

        q = BatchQueue()
        self.assertIsNone(q._next_queued())

    def test_next_queued_skips_done(self):
        from workers.batch_queue import BatchJob, BatchQueue

        q = BatchQueue()
        j1 = q.add("f1.wav")
        q.add("f2.wav")
        j1.status = BatchJob.DONE
        self.assertEqual(q._next_queued().file_path, "f2.wav")


if __name__ == "__main__":
    unittest.main()
