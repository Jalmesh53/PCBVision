import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path("/home/pi4/PCBVISION").resolve()

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ============================================================
# PCBVISION MODULES
# ============================================================

from main import PCBVision
from templates.template_builder import TemplateBuilder

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PCBVISION API",
    description="Raspberry Pi backend API for PCBVISION",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

inspection_engine = PCBVision()

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ============================================================
# HELPERS
# ============================================================

def validate_project_image(path: str) -> Path:
    """Only serve existing files located inside the project directory."""
    try:
        image_path = Path(path).resolve(strict=True)
        image_path.relative_to(BASE_DIR)
    except (ValueError, OSError, RuntimeError):
        raise HTTPException(
            status_code=404,
            detail="Image not found inside the PCBVISION project.",
        )

    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    return image_path


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "name": "PCBVISION",
        "status": "running",
        "version": "1.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "PCBVISION API",
    }


# ============================================================
# IMAGE SERVING
# ============================================================

@app.get("/inspection/image")
def get_inspection_image(path: str):
    """
    Serve inspection images to the Windows dashboard.

    Example:
    /inspection/image?path=/home/pi4/PCBVISION/data/output/detection_result.jpg
    """
    image_path = validate_project_image(path)
    return FileResponse(path=str(image_path))


# ============================================================
# CAMERA STATUS
# ============================================================

@app.get("/camera/status")
def camera_status():
    if not shutil.which("rpicam-still"):
        return {
            "connected": False,
            "status": "unavailable",
            "message": "rpicam-still command not found.",
        }

    return {
        "connected": True,
        "status": "available",
        "message": "Camera capture utility is available.",
    }


# ============================================================
# CAMERA CAPTURE
# ============================================================

@app.post("/camera/capture")
async def capture_camera():
    camera_command = shutil.which("rpicam-still")

    if not camera_command:
        raise HTTPException(
            status_code=503,
            detail="rpicam-still command not found.",
        )

    capture_dir = BASE_DIR / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)

    image_path = capture_dir / "latest_capture.jpg"

    command = [
        camera_command,
        "--output", str(image_path),
        "--timeout", "2000",
        "--nopreview",
    ]

    try:
        result = await run_in_threadpool(
            subprocess.run,
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="Camera capture timed out.",
        ) from exc

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=result.stderr.strip() or "Camera capture failed.",
        )

    if not image_path.is_file() or image_path.stat().st_size == 0:
        raise HTTPException(
            status_code=500,
            detail="Camera did not produce a valid image.",
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/jpeg",
        filename="pcb_capture.jpg",
    )


# ============================================================
# FULL INSPECTION - CAMERA
# ============================================================

@app.post("/inspection/run")
async def run_inspection():
    try:
        result = await run_in_threadpool(inspection_engine.inspect)

        if not isinstance(result, dict):
            raise RuntimeError("Inspection pipeline returned an invalid result.")

        return {
            "status": "success",
            "inspection": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inspection failed: {exc}",
        ) from exc


# ============================================================
# FULL INSPECTION - UPLOADED IMAGE
# ============================================================

@app.post("/inspection/upload")
async def inspect_uploaded_pcb(image: UploadFile = File(...)):
    suffix = Path(image.filename or "").suffix.lower()

    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Use JPG, JPEG, PNG, or BMP.",
        )

    input_dir = BASE_DIR / "data" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    image_path = input_dir / f"upload_{uuid.uuid4().hex[:12]}{suffix}"

    try:
        image_data = await image.read()

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        image_path.write_bytes(image_data)

        result = await run_in_threadpool(
            inspection_engine.inspect_uploaded_image,
            image_path,
        )

        if not isinstance(result, dict):
            raise RuntimeError("Inspection pipeline returned an invalid result.")

        return {
            "status": "success",
            "inspection": result,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Uploaded image inspection failed: {exc}",
        ) from exc
    finally:
        await image.close()


# ============================================================
# CREATE TEMPLATE
# ============================================================

@app.post("/templates/create")
async def create_template(
    template_name: str = Form(...),
    reference_images: list[UploadFile] = File(...),
):
    template_name = template_name.strip()

    if not template_name:
        raise HTTPException(
            status_code=400,
            detail="Template name cannot be empty.",
        )

    if not 4 <= len(reference_images) <= 5:
        raise HTTPException(
            status_code=400,
            detail="Upload between 4 and 5 reference images.",
        )

    try:
        with TemporaryDirectory(prefix="pcbvision_refs_") as temp_dir:
            image_paths = []

            for index, upload in enumerate(reference_images, start=1):
                suffix = Path(upload.filename or "").suffix.lower()

                if suffix not in ALLOWED_IMAGE_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported image type for file {index}.",
                    )

                content = await upload.read()

                if not content:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Reference image {index} is empty.",
                    )

                image_path = Path(temp_dir) / f"reference_{index}{suffix}"
                image_path.write_bytes(content)
                image_paths.append(str(image_path))

            builder = TemplateBuilder()

            result = await run_in_threadpool(
                builder.create_template,
                template_name=template_name,
                reference_images=image_paths,
            )

            if not isinstance(result, dict):
                raise RuntimeError("Template builder returned invalid data.")

            return {
                "status": "success",
                "template": result,
            }

    except HTTPException:
        raise
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Template creation failed: {exc}",
        ) from exc
    finally:
        for upload in reference_images:
            await upload.close()
