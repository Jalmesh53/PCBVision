import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path("/home/pi4/PCBVISION")
sys.path.insert(0, str(BASE_DIR))


# ============================================================
# IMPORTS
# ============================================================

from detection.detector import PCBDetector
from ocr.ocr import ComponentOCR

from verification.x_mccv import (
    XMCCV,
    calculate_orientation,
    detect_polarity,
    calculate_size_shape
)


# ============================================================
# COMPONENT CROP
# ============================================================

def crop_component(image_path, bbox, output_path):

    import cv2

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    x1 = max(0, int(bbox["x1"]))
    y1 = max(0, int(bbox["y1"]))
    x2 = min(image.shape[1], int(bbox["x2"]))
    y2 = min(image.shape[0], int(bbox["y2"]))

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        crop
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print(" PCBVISION DETECTOR → OCR → X-MCCV")
    print("========================================")

    # --------------------------------------------------------
    # Image used ONLY for this integration test
    # --------------------------------------------------------

    image_path = (
        BASE_DIR /
        "data" /
        "input" /
        "test.jpg"
    )

    if not image_path.exists():

        print(
            f"\nImage not found: {image_path}"
        )

        return

    # --------------------------------------------------------
    # 1. YOLO
    # --------------------------------------------------------

    print("\n[1] Running YOLO detection...")

    detector = PCBDetector()

    detections = detector.detect(
        image_path,
        confidence=0.25
    )

    if not detections:

        print("\nNo components detected.")

        return

    print(
        f"\nYOLO detections: "
        f"{len(detections)}"
    )

    # --------------------------------------------------------
    # 2. OCR
    # --------------------------------------------------------

    print("\n[2] Running OCR on detected components...")

    ocr_engine = ComponentOCR()

    crop_dir = (
        BASE_DIR /
        "data" /
        "output" /
        "component_crops"
    )

    for index, detection in enumerate(detections):

        crop_path = (
            crop_dir /
            f"component_{index + 1}.jpg"
        )

        crop_component(
            image_path,
            detection["bbox"],
            crop_path
        )

        if crop_path.exists():

            try:

                text = ocr_engine.extract_text(
                    crop_path
                )

            except Exception as error:

                print(
                    f"OCR error for component "
                    f"{index + 1}: {error}"
                )

                text = ""

        else:

            text = ""

        detection["ocr_text"] = text

        print(
            f"Component {index + 1}: "
            f"{detection['class_name']} "
            f"| OCR: "
            f"{text if text else '[none]'}"
        )

    # --------------------------------------------------------
    # 3. Additional X-MCCV features
    # --------------------------------------------------------

    print(
        "\n[3] Calculating X-MCCV features..."
    )

    for detection in detections:

        detection["orientation"] = (
            calculate_orientation(
                image_path,
                detection["bbox"]
            )
        )

        detection["polarity"] = (
            detect_polarity(
                image_path,
                detection["bbox"],
                detection["class_name"]
            )
        )

        detection["size_shape"] = (
            calculate_size_shape(
                detection
            )
        )

    # --------------------------------------------------------
    # 4. Temporary reference
    #
    # ONLY for testing module communication.
    # Final system will load this from the PCB template.
    # --------------------------------------------------------

    print(
        "\n[4] Creating temporary reference..."
    )

    expected_components = []

    for detection in detections:

        expected_components.append({

            "class_name":
                detection["class_name"],

            "expected_count":
                1,

            "expected_center": {

                "x":
                    detection[
                        "normalized_center"
                    ]["x"],

                "y":
                    detection[
                        "normalized_center"
                    ]["y"]
            },

            "expected_orientation":
                detection["orientation"],

            "polarity_required":
                False,

            "expected_polarity":
                None,

            "size_shape_required":
                True,

            "expected_size":
                detection["size_shape"],

            "spacing_required":
                False,

            "spacing_reference":
                None,

            "expected_spacing":
                None
        })

    # --------------------------------------------------------
    # 5. X-MCCV
    # --------------------------------------------------------

    print(
        "\n[5] Running X-MCCV..."
    )

    verifier = XMCCV(

        position_tolerance=0.10,

        orientation_tolerance=20,

        size_tolerance=0.20,

        spacing_tolerance=0.20
    )

    results = verifier.verify(
        detections,
        expected_components
    )

    # --------------------------------------------------------
    # 6. Results
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        " X-MCCV RESULT"
    )

    print(
        "========================================"
    )

    for result in results:

        print(
            f"\nComponent: "
            f"{result['class_name']}"
        )

        print(
            f"Presence: "
            f"{result['presence']['pass']}"
        )

        print(
            f"Count: "
            f"{result['count']['pass']}"
        )

        print(
            f"Position: "
            f"{result['position']['pass']}"
        )

        print(
            f"Identity: "
            f"{result['identity']['pass']}"
        )

        print(
            f"Orientation: "
            f"{result['orientation']['pass']}"
        )

        print(
            f"Polarity: "
            f"{result['polarity']['pass']}"
        )

        print(
            f"Size/Shape: "
            f"{result['size_shape']['pass']}"
        )

        print(
            f"Spacing: "
            f"{result['spacing']['pass']}"
        )

        print(
            f"OCR: "
            f"{result['identity']['ocr_text']}"
        )

        print(
            f"\nCOMPONENT RESULT: "
            f"{'PASS' if result['component_pass'] else 'FAIL'}"
        )

    print(
        "\n========================================"
    )

    print(
        " DETECTOR → OCR → X-MCCV OK"
    )

    print(
        "========================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
