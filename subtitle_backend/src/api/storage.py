import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any

# PUBLIC_INTERFACE
@dataclass
class UploadRecord:
    """Represents a user upload containing video and subtitle files."""
    upload_id: str
    video_ext: str
    subtitle_ext: str
    created_at: float
    # absolute paths for stored files
    video_path: str
    subtitle_path: str


# PUBLIC_INTERFACE
@dataclass
class JobRecord:
    """Represents processing job state and result mapping."""
    job_id: str
    upload_id: str
    status: str  # queued, running, done, error
    message: Optional[str]
    result_file: Optional[str]
    updated_at: float


class StorageError(Exception):
    """Generic storage error wrapper."""


# PUBLIC_INTERFACE
class StorageBackend:
    """Abstract interface for persistence."""

    def save_upload(self, video_src, video_ext: str, subtitle_src, subtitle_ext: str) -> UploadRecord:
        """Persist the uploaded files and return an UploadRecord."""
        raise NotImplementedError

    def get_upload(self, upload_id: str) -> Optional[UploadRecord]:
        """Retrieve a previously saved upload by ID."""
        raise NotImplementedError

    def create_job(self, upload_id: str) -> JobRecord:
        """Create a new job for an upload. Initially queued."""
        raise NotImplementedError

    def update_job(self, job_id: str, **fields) -> JobRecord:
        """Update fields on a job record and return updated record."""
        raise NotImplementedError

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Fetch a job record by ID."""
        raise NotImplementedError

    def save_result(self, job_id: str, src_path: str) -> str:
        """Move/copy result file to managed results location and return final absolute path."""
        raise NotImplementedError

    def cleanup(self, max_age_seconds: int) -> Dict[str, int]:
        """Remove aged uploads/results. Returns counts {'uploads': n, 'jobs': m, 'files': k}."""
        raise NotImplementedError

    # Utility for filesystem paths
    def get_paths(self) -> Dict[str, str]:
        """Return important base directories for inspection."""
        raise NotImplementedError


class InMemoryFileSystemStorage(StorageBackend):
    """
    Filesystem-based persistence with an in-memory index for upload and job metadata.
    Suitable for single-process deployments. Provides hooks to replace with DB later.
    """
    def __init__(self, base_dir: Optional[str] = None):
        # Set base directories
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.upload_dir = os.path.join(self.data_dir, "uploads")
        self.results_dir = os.path.join(self.data_dir, "results")
        self.previews_dir = os.path.join(self.data_dir, "previews")

        for d in [self.data_dir, self.upload_dir, self.results_dir, self.previews_dir]:
            os.makedirs(d, exist_ok=True)

        # In-memory indices
        self._uploads: Dict[str, UploadRecord] = {}
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.RLock()

    def get_paths(self) -> Dict[str, str]:
        return {
            "base_dir": self.base_dir,
            "data_dir": self.data_dir,
            "upload_dir": self.upload_dir,
            "results_dir": self.results_dir,
            "previews_dir": self.previews_dir,
        }

    def save_upload(self, video_src, video_ext: str, subtitle_src, subtitle_ext: str) -> UploadRecord:
        with self._lock:
            upload_id = str(uuid.uuid4())
            folder = os.path.join(self.upload_dir, upload_id)
            os.makedirs(folder, exist_ok=True)

            video_path = os.path.join(folder, f"video{video_ext}")
            subtitle_path = os.path.join(folder, f"subtitle{subtitle_ext}")

            try:
                # video_src / subtitle_src are file-like objects
                with open(video_path, "wb") as vf:
                    shutil.copyfileobj(video_src, vf)
                with open(subtitle_path, "wb") as sf:
                    shutil.copyfileobj(subtitle_src, sf)
            except Exception as e:
                # Cleanup partially written folder
                try:
                    shutil.rmtree(folder, ignore_errors=True)
                finally:
                    raise StorageError(f"Failed to save upload: {e}")

            rec = UploadRecord(
                upload_id=upload_id,
                video_ext=video_ext,
                subtitle_ext=subtitle_ext,
                created_at=time.time(),
                video_path=video_path,
                subtitle_path=subtitle_path,
            )
            self._uploads[upload_id] = rec
            return rec

    def get_upload(self, upload_id: str) -> Optional[UploadRecord]:
        with self._lock:
            return self._uploads.get(upload_id)

    def create_job(self, upload_id: str) -> JobRecord:
        with self._lock:
            if upload_id not in self._uploads:
                raise StorageError("upload_id not found")
            job_id = str(uuid.uuid4())
            rec = JobRecord(
                job_id=job_id,
                upload_id=upload_id,
                status="queued",
                message="Queued for processing",
                result_file=None,
                updated_at=time.time(),
            )
            self._jobs[job_id] = rec
            return rec

    def update_job(self, job_id: str, **fields) -> JobRecord:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                raise StorageError("job_id not found")
            for k, v in fields.items():
                if hasattr(rec, k):
                    setattr(rec, k, v)
            rec.updated_at = time.time()
            self._jobs[job_id] = rec
            return rec

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def save_result(self, job_id: str, src_path: str) -> str:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise StorageError("job_id not found")
            dest_folder = os.path.join(self.results_dir, job_id)
            os.makedirs(dest_folder, exist_ok=True)
            dest_path = os.path.join(dest_folder, os.path.basename(src_path))
            shutil.move(src_path, dest_path)
            job.result_file = dest_path
            job.status = "done"
            job.message = "Processing completed"
            job.updated_at = time.time()
            self._jobs[job_id] = job
            return dest_path

    def cleanup(self, max_age_seconds: int) -> Dict[str, int]:
        now = time.time()
        removed_uploads = 0
        removed_jobs = 0
        removed_files = 0

        # Remove old uploads (folder + index) when age exceeded and not referenced by any pending job
        with self._lock:
            # Identify referenced uploads
            referenced_uploads = {job.upload_id for job in self._jobs.values() if job.status in ("queued", "running")}
            for upload_id, rec in list(self._uploads.items()):
                if upload_id in referenced_uploads:
                    continue
                if now - rec.created_at >= max_age_seconds:
                    folder = os.path.dirname(rec.video_path)
                    if os.path.isdir(folder):
                        try:
                            removed_files += self._count_files(folder)
                            shutil.rmtree(folder, ignore_errors=True)
                        except Exception:
                            pass
                    self._uploads.pop(upload_id, None)
                    removed_uploads += 1

            # Remove old results/job indices where done/error long ago
            for job_id, job in list(self._jobs.items()):
                if now - job.updated_at >= max_age_seconds:
                    # delete result folder if present
                    if job.result_file:
                        job_folder = os.path.dirname(job.result_file)
                        if os.path.isdir(job_folder):
                            try:
                                removed_files += self._count_files(job_folder)
                                shutil.rmtree(job_folder, ignore_errors=True)
                            except Exception:
                                pass
                    self._jobs.pop(job_id, None)
                    removed_jobs += 1

        return {"uploads": removed_uploads, "jobs": removed_jobs, "files": removed_files}

    @staticmethod
    def _count_files(folder: str) -> int:
        count = 0
        for _, _, files in os.walk(folder):
            count += len(files)
        return count


# PUBLIC_INTERFACE
def make_storage() -> StorageBackend:
    """
    Factory to create the current storage backend.
    Future: read env DB_URL, etc., and return a DB-backed storage.
    """
    # Placeholder for future DB integration; currently filesystem-based:
    return InMemoryFileSystemStorage()


# PUBLIC_INTERFACE
def subtitle_metadata(path: str) -> Dict[str, Any]:
    """Basic subtitle metadata: extension, format, line count (approx), size bytes."""
    ext = os.path.splitext(path)[1].lower()
    fmt = {
        ".srt": "SubRip",
        ".ass": "Advanced SubStation Alpha",
        ".ssa": "SubStation Alpha",
        ".vtt": "WebVTT",
    }.get(ext, "unknown")
    try:
        size = os.path.getsize(path)
    except Exception:
        size = None
    # Approximate line count: count cue/dialogue lines
    line_count = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                # heuristic to count useful lines
                if ext in (".ass", ".ssa"):
                    if line.startswith("Dialogue:"):
                        line_count += 1
                elif ext == ".vtt":
                    if "-->" in line:
                        line_count += 1
                else:
                    # srt: count numeric sequence or time lines
                    if "-->" in line:
                        line_count += 1
    except Exception:
        line_count = None

    return {"extension": ext, "format": fmt, "size_bytes": size, "line_count": line_count}
