import subprocess
from pathlib import Path
from datetime import datetime
from PIL import Image


BASE_DIR = Path("/home/pi4/PCBVISION")
INPUT_DIR = BASE_DIR / "data" / "input"


def capture_image():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"pcb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    original_path = INPUT_DIR / f"original_{filename}"
    final_path = INPUT_DIR / filename

    command = [
        "rpicam-still",
        "-o", str(original_path),
        "--autofocus-mode", "continuous",
        "--roi", "0.15,0.15,0.70,0.70",
        "--nopreview"
    ]

    try:
        subprocess.run(command, check=True)

        # Rotate image 180° after capture
        image = Image.open(original_path)
        image = image.rotate(180, expand=True)
        image.save(final_path)

        # Remove temporary original image
        original_path.unlink()

        print(f"Image captured: {final_path}")
        return str(final_path)

    except subprocess.CalledProcessError as e:
        print(f"Camera capture failed: {e}")
        return None

    except Exception as e:
        print(f"Image processing failed: {e}")
        return None


if __name__ == "__main__":
    capture_image()
