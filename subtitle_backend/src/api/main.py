from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import os
import shutil
import uuid
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
except Exception:
    # Script context: adjust sys.path then use absolute import
    import sys

    current_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from src.api.processing import process_subtitle  # type: ignore

# FastAPI metadata and tags for OpenAPI
app = FastAPI(
    title="Subtitle Repositioning Backend",
    description="APIs to upload video & subtitle, run OCR-based repositioning to avoid overlap with burnt-in text, preview results, and download processed files.",
    version="0.1.0",
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

# Storage directories
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PREVIEW_DIR = os.path.join(DATA_DIR, "previews")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# In-memory job registry for simplicity
JOBS = {}


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
    return {"message": "Healthy"}


# PUBLIC_INTERFACE
@app.post(
    "/upload",
    tags=["upload"],
    summary="Upload video and subtitle",
    description="Upload a video file and a subtitle file to start a processing session. Returns an upload_id that can be used to start processing.",
    responses={
        200: {"description": "Upload successful"},
        400: {"description": "Invalid input"},
    },
)
async def upload_files(video: UploadFile = File(...), subtitle: UploadFile = File(...)):
    # Basic validation
    allowed_video_ext = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
    allowed_sub_ext = {".srt", ".ass", ".ssa", ".vtt"}

    vext = os.path.splitext(video.filename or "")[1].lower()
    sext = os.path.splitext(subtitle.filename or "")[1].lower()

    if vext not in allowed_video_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {vext}")
    if sext not in allowed_sub_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported subtitle type: {sext}")

    upload_id = str(uuid.uuid4())
    upload_folder = os.path.join(UPLOAD_DIR, upload_id)
    os.makedirs(upload_folder, exist_ok=True)

    video_path = os.path.join(upload_folder, f"video{vext}")
    subtitle_path = os.path.join(upload_folder, f"subtitle{sext}")

    # Save files
    try:
        with open(video_path, "wb") as vf:
            shutil.copyfileobj(video.file, vf)
        with open(subtitle_path, "wb") as sf:
            shutil.copyfileobj(subtitle.file, sf)
    finally:
        await video.close()
        await subtitle.close()

    return {"upload_id": upload_id, "video_path": video_path, "subtitle_path": subtitle_path}


def _run_processing(job_id: str, upload_id: str, max_workers: int):
    try:
        JOBS[job_id] = {"status": "running", "message": "Processing started", "result_file": None}
        upload_folder = os.path.join(UPLOAD_DIR, upload_id)
        # Discover saved files
        candidates = os.listdir(upload_folder)
        video_path = next((os.path.join(upload_folder, f) for f in candidates if os.path.splitext(f)[0] == "video"), None)
        sub_path = next((os.path.join(upload_folder, f) for f in candidates if os.path.splitext(f)[0] == "subtitle"), None)

        if not video_path or not sub_path:
            raise RuntimeError("Uploaded files not found for given upload_id")

        # Run processing; returns path of output file (in same folder as subtitle)
        output_file = process_subtitle(video_path, sub_path, max_workers=max_workers)

        # Move result into results dir under job-id for download
        os.makedirs(os.path.join(RESULTS_DIR, job_id), exist_ok=True)
        result_path = os.path.join(RESULTS_DIR, job_id, os.path.basename(output_file))
        shutil.move(output_file, result_path)

        JOBS[job_id] = {"status": "done", "message": "Processing completed", "result_file": result_path}
    except Exception as e:
        JOBS[job_id] = {"status": "error", "message": f"{e}\n{traceback.format_exc()}", "result_file": None}


# PUBLIC_INTERFACE
@app.post(
    "/process",
    tags=["process"],
    summary="Start processing",
    description="Start the repositioning process for a previously uploaded video and subtitle. Returns a job_id to monitor status.",
    response_model=ProcessResponse,
)
async def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    upload_folder = os.path.join(UPLOAD_DIR, req.upload_id)
    if not os.path.exists(upload_folder):
        raise HTTPException(status_code=404, detail="upload_id not found")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "message": "Queued for processing", "result_file": None}

    # Launch background processing
    background_tasks.add_task(_run_processing, job_id=job_id, upload_id=req.upload_id, max_workers=req.max_workers or 10)
    return ProcessResponse(job_id=job_id)


# PUBLIC_INTERFACE
@app.get(
    "/status/{job_id}",
    tags=["process"],
    summary="Get job status",
    description="Check the current processing status for a job.",
    response_model=JobStatusResponse,
)
def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job_id not found")
    job = JOBS[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        message=job.get("message"),
        result_file=os.path.basename(job["result_file"]) if job.get("result_file") else None,
    )


# PUBLIC_INTERFACE
@app.get(
    "/preview/{job_id}",
    tags=["results"],
    summary="Preview processed subtitle file (JSON info)",
    description="Returns a lightweight preview of the processed file info. For full file contents, use the download endpoint.",
)
def preview_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job_id not found")
    job = JOBS[job_id]
    if job.get("status") != "done" or not job.get("result_file"):
        return JSONResponse({"ready": False, "message": job.get("message", "Not completed")})
    result_path = job["result_file"]
    size = os.path.getsize(result_path)
    name = os.path.basename(result_path)
    return {"ready": True, "file_name": name, "size_bytes": size}


# PUBLIC_INTERFACE
@app.get(
    "/download/{job_id}",
    tags=["results"],
    summary="Download processed subtitle file",
    description="Downloads the processed subtitle file created by the job.",
)
def download_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job_id not found")
    job = JOBS[job_id]
    if job.get("status") != "done" or not job.get("result_file"):
        raise HTTPException(status_code=409, detail="Result is not ready")
    file_path = job["result_file"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result file missing")
    return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type="application/octet-stream")


if __name__ == "__main__":
    # Allow running directly: python subtitle_backend/src/api/main.py
    # Determine project root to run uvicorn with the correct app dir
    import uvicorn

    current_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    # Run uvicorn programmatically; host/port can be customized via env if needed
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False, app_dir=src_dir)  # type: ignore
