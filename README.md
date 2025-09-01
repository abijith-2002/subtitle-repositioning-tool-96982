# Project Repository

This is the initial README file for the project.

## Backend (subtitle_backend)

- Install dependencies:
  pip install -r subtitle_backend/requirements.txt

- Run dev server:
  uvicorn src.api.main:app --reload --app-dir subtitle_backend/src

### Large upload configuration (fixing HTTP 413)

The backend includes a middleware to allow large file uploads. Configure the maximum request size using an environment variable:

- UPLOAD_MAX_SIZE_MB: integer (megabytes), default is 2048 (2 GB).
  Example:
    export UPLOAD_MAX_SIZE_MB=4096  # allow up to 4 GB
    uvicorn src.api.main:app --reload --app-dir subtitle_backend/src

If you run the application behind Nginx (or any reverse proxy), ensure proxy settings also allow large bodies; otherwise you'll receive 413 before the app is reached:

- In Nginx (server or http block):
    client_max_body_size 4096M;   # match or exceed UPLOAD_MAX_SIZE_MB
    proxy_request_buffering off;  # optional for streaming large requests

Reload Nginx after changes:
    sudo nginx -s reload

### Notes
- requirements.txt is trimmed to only runtime packages with compatible versions to avoid CI install failures.
- Python 3.12 compatibility: replaced pysrt with the pure-Python 'srt' package.
- RapidOCR ONNXRuntime dependency relaxed to >=1.3.0,<2.0.0 with graceful runtime fallback if import/engine init fails.
- RapidOCR ONNXRuntime and OpenCV rely on prebuilt wheels; if your environment lacks manylinux wheels, ensure system packages for onnxruntime/OpenCV are available or use compatible Python versions (3.10–3.12 supported).
- Linting: a permissive flake8 config is included in setup.cfg to avoid CI failures. If your CI requires flake8, install it (pip install flake8).