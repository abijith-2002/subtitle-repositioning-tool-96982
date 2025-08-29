# Project Repository

This is the initial README file for the project.

## Backend (subtitle_backend)

- Install dependencies:
  pip install -r subtitle_backend/requirements.txt

- Run dev server:
  uvicorn src.api.main:app --reload --app-dir subtitle_backend/src

Notes:
- requirements.txt is trimmed to only runtime packages with compatible versions to avoid CI install failures.
- RapidOCR ONNXRuntime and OpenCV rely on prebuilt wheels; if your environment lacks manylinux wheels, ensure system packages for onnxruntime/OpenCV are available or use compatible Python versions (3.10–3.11 recommended).