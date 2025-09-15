# Project Repository

This is the initial README file for the project.

## Backend (subtitle_backend)

- Install dependencies:
  pip install -r subtitle_backend/requirements.txt

- Run dev server:
  uvicorn src.api.main:app --reload --app-dir subtitle_backend/src

### Large upload configuration (fixing HTTP 413)

The backend enforces a maximum request body size via middleware. Configure with:

- UPLOAD_MAX_SIZE_MB: integer (megabytes), default: 2048 (2 GB)
  Example:
    export UPLOAD_MAX_SIZE_MB=4096  # allow up to 4 GB
    uvicorn src.api.main:app --reload --app-dir subtitle_backend/src

If behind a reverse proxy (e.g., Nginx), also update proxy limits:

- In Nginx (server/http block):
    client_max_body_size 4096M;
    proxy_request_buffering off;

Reload Nginx after changes:
    sudo nginx -s reload

### Storage design and persistence

A new storage abstraction layer manages uploads and processing jobs:

- src/api/storage.py provides a StorageBackend interface and an InMemoryFileSystemStorage implementation.
- Default behavior: in-memory indices with filesystem persistence:
  - data/uploads/{upload_id}/video.ext and subtitle.ext
  - data/results/{job_id}/processed_output.ext
- Future database integration:
  - Replace make_storage() to return a DB-backed implementation.
  - Keep file artifacts in filesystem/object storage and index records in DB.

Environment variables for storage/cleanup:

- ENABLE_CLEANUP: "true" to enable background cleanup (default "false")
- FILE_RETENTION_SECONDS: seconds to retain uploads/jobs (default 86400)
- CLEANUP_INTERVAL_SECONDS: interval to run cleanup checks (default 600)
- OCR_MAX_CONCURRENCY: cap concurrent OCR-heavy processing (default 2)

### API flow

1. POST /upload
   - Form-data: video (binary), subtitle (binary)
   - Returns: { ok, upload_id, video_name, subtitle_name }

2. POST /process
   - JSON: { upload_id, max_workers? }
   - Returns: { job_id }

3. GET /status/{job_id}
   - Returns job status; when done includes result_file token (basename).

4. GET /preview/{job_id}
   - Returns { ready, file_name, extension, format, size_bytes, line_count } when ready.

5. GET /download/{job_id}
   - Downloads the processed subtitle file.

### Error handling and response consistency

- Errors return JSON { ok: false, error: { code, message, details? } } for upload endpoint.
- Other endpoints use standard HTTPException with clear messages.
- Background job failures surface via status endpoint with error message.

### OCR concurrency and fallback

- RapidOCR ONNXRuntime is optional; if import fails at runtime, OCR gracefully skips detections and positions default accordingly.
- OCR_MAX_CONCURRENCY limits simultaneous OCR workloads for stability on constrained hosts.

### OpenAPI

- To regenerate API spec, run:
  PYTHONPATH=subtitle_backend/src python subtitle_backend/src/api/generate_openapi.py
- Updated schema is written to subtitle_backend/interfaces/openapi.json

### Dependencies

- Use subtitle_backend/requirements.txt (single source of runtime dependencies).
- Removed redundant src/api/requirements.txt (was causing confusion).
- Python 3.12 compatibility: uses pure-Python 'srt' package for SRT parsing.
- If CI requires linting, flake8 is already included.
