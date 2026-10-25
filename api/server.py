
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path("/home/pi4/PCBVISION")

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ============================================================
# PCBVISION MODULES
# ============================================================

from main import PCBVision
from templates.template_builder import TemplateBuilder
from templates.templates import TemplateManager
from camera.camera import capture_image

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

# ============================================================
# PCBVISION INSPECTION ENGINE
# ============================================================

inspection_engine = PCBVision()
template_manager = TemplateManager()

# In-memory capture sessions. A session is not a saved template.
template_sessions = {}

TEMPLATE_SESSION_DIR = BASE_DIR / "data" / "template_sessions"
TEMPLATE_SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# REQUEST MODELS
# ============================================================

class TemplateStartRequest(BaseModel):
    template_name: str
    expected_images: int


class TemplateSessionRequest(BaseModel):
    session_id: str


# ============================================================
# ROOT AND HEALTH
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

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def validate_project_image(path: str) -> Path:
    try:
        image_path = Path(path).resolve(strict=True)
        image_path.relative_to(BASE_DIR)
    except (ValueError, OSError, RuntimeError):
        raise HTTPException(
            status_code=404,
            detail="Image not found inside the PCBVISION project.",
        )

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image not found.",
        )

    if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format.",
        )

    return image_path


@app.get("/inspection/image")
def get_inspection_image(path: str):
    image_path = validate_project_image(path)

    return FileResponse(
        path=str(image_path),
        media_type="image/jpeg",
    )


# ============================================================
# CAMERA STATUS
# ============================================================

@app.get("/camera/status")
def camera_status():
    camera_command = shutil.which("rpicam-still")

    if not camera_command:
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
        "--output",
        str(image_path),
        "--timeout",
        "2000",
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
            detail=(
                "Camera capture failed: "
                f"{result.stderr.strip() or 'Unknown camera error.'}"
            ),
        )

    if not image_path.exists() or image_path.stat().st_size == 0:
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
# FULL PCB INSPECTION - CAMERA
# ============================================================

@app.post("/inspection/run")
async def run_inspection():
    try:
        result = await run_in_threadpool(
            inspection_engine.inspect
        )

        if not isinstance(result, dict):
            raise RuntimeError(
                "Inspection pipeline did not return a valid result."
            )

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
# FULL PCB INSPECTION - UPLOADED IMAGE
# ============================================================

@app.post("/inspection/upload")
async def inspect_uploaded_pcb(
    image: UploadFile = File(...),
):
    suffix = Path(image.filename or "").suffix.lower()

    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format.",
        )

    input_dir = BASE_DIR / "data" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    image_path = (
        input_dir / f"upload_{uuid.uuid4().hex[:12]}{suffix}"
    )

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
            raise RuntimeError(
                "Inspection pipeline did not return a valid result."
            )

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
# TEMPLATE SESSION HELPERS
# ============================================================

def get_template_session(session_id: str):
    session = template_sessions.get(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Template session not found or already closed.",
        )

    return session


# ============================================================
# LIST SAVED TEMPLATES
# ============================================================

@app.get("/templates")
def list_templates():
    return {
        "status": "success",
        "templates": template_manager.list_templates(),
    }


# ============================================================
# START TEMPLATE CAPTURE SESSION
# ============================================================

@app.post("/templates/start")
async def start_template_session(
    request: TemplateStartRequest,
):
    template_name = request.template_name.strip()

    if not template_name:
        raise HTTPException(
            status_code=400,
            detail="Template name cannot be empty.",
        )

    if not 4 <= request.expected_images <= 5:
        raise HTTPException(
            status_code=400,
            detail="Reference image count must be between 4 and 5.",
        )

    try:
        safe_name = template_manager._sanitize_name(template_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if safe_name in template_manager.list_templates():
        raise HTTPException(
            status_code=409,
            detail=f"Template already exists: {safe_name}",
        )

    # Prevent two unfinished sessions for the same template name.
    for existing in template_sessions.values():
        if existing["template_name"] == safe_name:
            raise HTTPException(
                status_code=409,
                detail="A capture session already exists for this template.",
            )

    session_id = uuid.uuid4().hex
    session_dir = TEMPLATE_SESSION_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=False)

    template_sessions[session_id] = {
        "template_name": safe_name,
        "expected_images": request.expected_images,
        "image_paths": [],
        "session_dir": session_dir,
    }

    inspection_engine.hardware.template_capture_progress(
        safe_name,
        0,
        request.expected_images,
    )

    return {
        "status": "success",
        "session_id": session_id,
        "template_name": safe_name,
        "expected_images": request.expected_images,
        "captured_images": 0,
        "message": "Template session started. Use Next to capture the first image.",
    }


# ============================================================
# CAPTURE EXACTLY ONE TEMPLATE REFERENCE IMAGE
# ============================================================

@app.post("/templates/next")
async def capture_next_template_image(
    request: TemplateSessionRequest,
):
    session = get_template_session(request.session_id)

    captured_count = len(session["image_paths"])
    expected_count = session["expected_images"]

    if captured_count >= expected_count:
        raise HTTPException(
            status_code=409,
            detail="All required reference images have already been captured.",
        )

    next_index = captured_count + 1

    try:
        inspection_engine.hardware.template_capture_progress(
            session["template_name"],
            next_index,
            expected_count,
        )

        captured_path = await run_in_threadpool(capture_image)

        if not captured_path:
            raise RuntimeError("Camera capture failed.")

        captured_path = Path(captured_path)

        if not captured_path.is_file() or captured_path.stat().st_size == 0:
            raise RuntimeError("Camera returned an invalid image.")

        # Copy into a session-specific location so each reference
        # remains available independently until template creation.
        extension = captured_path.suffix.lower() or ".jpg"
        destination = (
            session["session_dir"]
            / f"reference_{next_index:02d}{extension}"
        )

        await run_in_threadpool(
            shutil.copy2,
            captured_path,
            destination,
        )

        session["image_paths"].append(str(destination))

        inspection_engine.hardware.image_captured()

        return {
            "status": "success",
            "session_id": request.session_id,
            "template_name": session["template_name"],
            "captured_images": len(session["image_paths"]),
            "expected_images": expected_count,
            "image_path": str(destination),
            "has_more_images": len(session["image_paths"]) < expected_count,
            "can_finish": len(session["image_paths"]) == expected_count,
            "message": (
                f"Captured image {len(session['image_paths'])}/{expected_count}."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        inspection_engine.hardware.error("CAPTURE ERROR")

        raise HTTPException(
            status_code=500,
            detail=f"Template image capture failed: {exc}",
        ) from exc


# ============================================================
# FINISH AND SAVE TEMPLATE
# ============================================================

@app.post("/templates/finish")
async def finish_template_session(
    request: TemplateSessionRequest,
):
    session = get_template_session(request.session_id)

    captured_count = len(session["image_paths"])
    expected_count = session["expected_images"]

    if captured_count != expected_count:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Capture all {expected_count} images before finishing. "
                f"Captured: {captured_count}."
            ),
        )

    # Recheck immediately before building.
    if session["template_name"] in template_manager.list_templates():
        raise HTTPException(
            status_code=409,
            detail="A template with this name already exists.",
        )

    try:
        inspection_engine.hardware.template_building()

        builder = TemplateBuilder()

        result = await run_in_threadpool(
            builder.create_template,
            template_name=session["template_name"],
            reference_images=session["image_paths"],
        )

        if not isinstance(result, dict):
            raise RuntimeError(
                "Template builder did not return template data."
            )

        inspection_engine.hardware.template_saved(
            session["template_name"]
        )

        # Close the session only after successful template creation.
        template_sessions.pop(request.session_id, None)
        shutil.rmtree(session["session_dir"], ignore_errors=True)

        return {
            "status": "success",
            "template": result,
            "template_name": session["template_name"],
            "reference_image_count": captured_count,
            "message": "Template created and saved successfully.",
        }

    except FileExistsError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        inspection_engine.hardware.error("TEMPLATE ERROR")

        raise HTTPException(
            status_code=500,
            detail=f"Template creation failed: {exc}",
        ) from exc


# ============================================================
# CANCEL TEMPLATE CAPTURE SESSION
# ============================================================

@app.post("/templates/cancel")
async def cancel_template_session(
    request: TemplateSessionRequest,
):
    session = get_template_session(request.session_id)

    template_sessions.pop(request.session_id, None)
    shutil.rmtree(session["session_dir"], ignore_errors=True)

    inspection_engine.hardware.reset()

    return {
        "status": "success",
        "message": "Template capture session cancelled.",
        "template_name": session["template_name"],
        "captured_images_discarded": len(session["image_paths"]),
    }


# ============================================================
# LEGACY TEMPLATE CREATE ENDPOINT
# ============================================================

@app.post("/templates/create")
async def create_template(
    template_name: str = Form(...),
    expected_images: int = Form(...),
):
    """
    Legacy endpoint: captures all reference images in one request.
    The dashboard should use /templates/start, /templates/next,
    and /templates/finish for step-by-step capture.
    """

    template_name = template_name.strip()

    if not template_name:
        raise HTTPException(
            status_code=400,
            detail="Template name cannot be empty.",
        )

    if not 4 <= expected_images <= 5:
        raise HTTPException(
            status_code=400,
            detail="Reference image count must be between 4 and 5.",
        )

    try:
        image_paths = []

        for index in range(1, expected_images + 1):
            inspection_engine.hardware.template_capture_progress(
                template_name,
                index,
                expected_images,
            )

            image_path = await run_in_threadpool(capture_image)

            if not image_path:
                raise RuntimeError(
                    f"Camera capture failed for reference image {index}."
                )

            image_path = Path(image_path)

            if not image_path.exists() or image_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Invalid captured image for reference image {index}."
                )

            image_paths.append(str(image_path))
            inspection_engine.hardware.image_captured()

        inspection_engine.hardware.template_building()

        builder = TemplateBuilder()

        result = await run_in_threadpool(
            builder.create_template,
            template_name=template_name,
            reference_images=image_paths,
        )

        inspection_engine.hardware.template_saved(template_name)

        return {
            "status": "success",
            "template": result,
            "reference_images": image_paths,
        }

    except HTTPException:
        raise

    except FileExistsError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        inspection_engine.hardware.error("TEMPLATE ERROR")

        raise HTTPException(
            status_code=500,
            detail=f"Template creation failed: {exc}",
        ) from exc
