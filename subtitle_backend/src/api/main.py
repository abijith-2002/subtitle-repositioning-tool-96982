from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Optional
import os
import traceback

"""
FastAPI application entrypoint.

Supports two startup modes:
1) Package mode (recommended): uvicorn src.api.main:app --app-dir subtitle_backend/src --reload
2) Script mode: python subtitle_backend/src/api/main.py
   In script mode we adjust sys.path and fall back to absolute imports to avoid relative import issues.
"""

# Import processing with dual strategy: relative (package) and absolute (script)
try:
    # Package context: "src.api" is a package and relative import is valid
    from .processing import process_subtitle  # type: ignore
    from .storage import make_storage, subtitle_metadata  # type: ignore
    from .utils import error_response, OcrConcurrencyLimiter, CleanupController  # type: ignore
except Exception:
    # Script context: adjust sys.path then use absolute import
    import sys
    current_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from src.api.processing import process_subtitle  # type: ignore
    from src.api.storage import make_storage, subtitle_metadata  # type: ignore
    from src.api.utils import error_response, OcrConcurrencyLimiter, CleanupController  # type: ignore

# FastAPI metadata and tags for OpenAPI
app = FastAPI(
    title="Subtitle Repositioning Backend",
    description="APIs to upload video & subtitle, run OCR-based repositioning to avoid overlap with burnt-in text, preview results, and download processed files.",
    version="0.2.0",
    openapi_tags=[
        {"name": "health", "description": "Health and status"},
        {"name": "upload", "description": "Upload video and subtitle files"},
        {"name": "process", "description": "Run repositioning job"},
        {"name": "results", "description": "Preview and download processed outputs"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class _BodySizeLimitMiddleware:
    """
    Middleware to enforce a maximum request body size.

    This helps avoid 413 errors by making the limit explicit and configurable via environment variables.

    Env:
      - UPLOAD_MAX_SIZE_MB: integer megabytes allowed for request body size; default 2048 (2 GB)
    """
    def __init__(self, app: FastAPI, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        if scope["type"] == "http":
            # Inspect content-length if provided to avoid reading the whole body
            headers = dict((k.decode("latin1"), v.decode("latin1")) for k, v in scope.get("headers", []))
            content_length = headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > self.max_body_bytes:
                        response = PlainTextResponse(
                            "Request entity too large",
                            status_code=413,
                            headers={"Connection": "close"},
                        )
                        return await response(scope, receive, send)
                except Exception:
                    # Ignore malformed content-length and fall back to streaming checks
                    pass

        # Fallback: pass-through to app; python-multipart streams to disk for file uploads.
        return await self.app(scope, receive, send)

# Configure maximum upload size from environment (in MB), default to 2048 MB
def _get_max_upload_bytes() -> int:
    try:
        mb = int(os.getenv("UPLOAD_MAX_SIZE_MB", "2048"))
        return max(1, mb) * 1024 * 1024
    except Exception:
        return 2048 * 1024 * 1024

# Install middleware
app.add_middleware(_BodySizeLimitMiddleware, max_body_bytes=_get_max_upload_bytes())

# Initialize storage and utilities
_storage = make_storage()
_ocr_limit = OcrConcurrencyLimiter()

# Background cleanup controller (optional)
_enable_cleanup = str(os.getenv("ENABLE_CLEANUP", "false")).lower() == "true"
_cleanup = CleanupController(_storage, enable=_enable_cleanup)

@app.on_event("startup")
def _on_startup():
    """Start background cleanup if enabled."""
    _cleanup.start()

@app.on_event("shutdown")
def _on_shutdown():
    """Stop background cleanup thread gracefully."""
    _cleanup.stop()


class ProcessRequest(BaseModel):
    # PUBLIC_INTERFACE
    """Request to process a given upload."""
    upload_id: str = Field(..., description="The upload identifier returned from the upload endpoint")
    max_workers: Optional[int] = Field(10, description="Maximum worker threads to use for processing")


class ProcessResponse(BaseModel):
    # PUBLIC_INTERFACE
    """Response providing the job identifier."""
    job_id: str = Field(..., description="Identifier for the processing job")


class JobStatusResponse(BaseModel):
    # PUBLIC_INTERFACE
    """Status of a processing job."""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current status: queued, running, done, error")
    message: Optional[str] = Field(None, description="Optional message for details")
    result_file: Optional[str] = Field(None, description="Path token to download the result file when done")


@app.get("/", tags=["health"], summary="Health Check", description="Basic health endpoint to verify service is running.")
def health_check():
    return {"ok": True, "message": "Healthy"}


# PUBLIC_INTERFACE
@app.post(
    "/upload",
    tags=["upload"],
    summary="Upload video and subtitle",
    description="Upload a video file and a subtitle file to start a processing session. Returns an upload_id that can be used to start processing. Maximum request size is configurable via the UPLOAD_MAX_SIZE_MB environment variable (default 2048 MB).",
    responses={
        200: {"description": "Upload successful"},
        400: {"description": "Invalid input"},
        413: {"description": "Uploaded files exceed configured size limit"},
    },
)
async def upload_files(video: UploadFile = File(...), subtitle: UploadFile = File(...)):
    """
    Upload a video and a subtitle file.

    Returns:
      - JSON { ok, upload_id, video_name, subtitle_name }
    Errors:
      - 400 on invalid extension
      - 500 on storage failure
    """
    allowed_video_ext = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
    allowed_sub_ext = {".srt", ".ass", ".ssa", ".vtt"}

    vext = os.path.splitext(video.filename or "")[1].lower()
    sext = os.path.splitext(subtitle.filename or "")[1].lower()

    if vext not in allowed_video_ext:
        return error_response(400, "unsupported_video_type", f"Unsupported video type: {vext}")
    if sext not in allowed_sub_ext:
        return error_response(400, "unsupported_subtitle_type", f"Unsupported subtitle type: {sext}")

    try:
        rec = _storage.save_upload(video.file, vext, subtitle.file, sext)
    finally:
        # Ensure temp file handles are closed
        await video.close()
        await subtitle.close()

    return {"ok": True, "upload_id": rec.upload_id, "video_name": video.filename, "subtitle_name": subtitle.filename}


def _run_processing(job_id: str, upload_id: str, max_workers: int):
    """
    Background processing function:
      - updates job state to running
      - runs processing with OCR concurrency limiter
      - saves result in storage and marks as done
      - marks error state on failure
    """
    try:
        _storage.update_job(job_id, status="running", message="Processing started")
        upload_rec = _storage.get_upload(upload_id)
        if not upload_rec:
            raise RuntimeError("Uploaded files not found for given upload_id")

        # Limit OCR concurrency for environments with limited resources
        with _ocr_limit.acquire():
            output_file = process_subtitle(upload_rec.video_path, upload_rec.subtitle_path, max_workers=max_workers)

        # Move result into managed results dir
        final_path = _storage.save_result(job_id, output_file)
        _storage.update_job(job_id, status="done", message="Processing completed", result_file=final_path)
    except Exception as e:
        _storage.update_job(job_id, status="error", message=f"{e}\n{traceback.format_exc()}", result_file=None)


# PUBLIC_INTERFACE
@app.post(
    "/process",
    tags=["process"],
    summary="Start processing",
    description="Start the repositioning process for a previously uploaded video and subtitle. Returns a job_id to monitor status.",
    response_model=ProcessResponse,
)
async def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Start processing for an upload.

    Body:
      - upload_id: string from /upload
      - max_workers: optional int, caps processing thread pool (default 10)

    Returns:
      - job_id
    Errors:
      - 404 if upload not found
    """
    upload = _storage.get_upload(req.upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="upload_id not found")

    # Create job record
    job = _storage.create_job(req.upload_id)

    # Launch background processing
    background_tasks.add_task(_run_processing, job_id=job.job_id, upload_id=req.upload_id, max_workers=req.max_workers or 10)
    return ProcessResponse(job_id=job.job_id)


# PUBLIC_INTERFACE
@app.get(
    "/status/{job_id}",
    tags=["process"],
    summary="Get job status",
    description="Check the current processing status for a job.",
    response_model=JobStatusResponse,
)
def get_status(job_id: str):
    """
    Path params:
      - job_id: returned from /process

    Returns:
      - JobStatusResponse with optional result_file token (basename).
    """
    job = _storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        message=job.message,
        result_file=os.path.basename(job.result_file) if job.result_file else None,
    )


# PUBLIC_INTERFACE
@app.get(
    "/preview/{job_id}",
    tags=["results"],
    summary="Preview processed subtitle file (JSON info)",
    description="Returns a lightweight preview with metadata (format, line_count, size_bytes). For full file contents, use the download endpoint.",
)
def preview_result(job_id: str):
    """
    Returns:
      - { ready: bool, file_name, size_bytes, format, line_count }
    Errors:
      - 404 if job not found
    """
    job = _storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    if job.status != "done" or not job.result_file:
        return JSONResponse({"ready": False, "message": job.message or "Not completed"})

    result_path = job.result_file
    meta = subtitle_metadata(result_path)
    name = os.path.basename(result_path)
    return {"ready": True, "file_name": name, **meta}


# PUBLIC_INTERFACE
@app.get(
    "/download/{job_id}",
    tags=["results"],
    summary="Download processed subtitle file",
    description="Downloads the processed subtitle file created by the job.",
)
def download_result(job_id: str):
    """
    Path params:
      - job_id

    Returns:
      - FileResponse download of the processed subtitle
    Errors:
      - 404 if job missing or file missing
      - 409 if not ready
    """
    job = _storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    if job.status != "done" or not job.result_file:
        raise HTTPException(status_code=409, detail="Result is not ready")
    file_path = job.result_file
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result file missing")
    return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type="application/octet-stream")


if __name__ == "__main__":
    # Allow running directly: python subtitle_backend/src/api/main.py
    import uvicorn
    current_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False, app_dir=src_dir)  # type: ignore
