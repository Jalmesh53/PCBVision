from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path("/home/pi4/PCBVISION")
MODEL_PATH = BASE_DIR / "models" / "best.pt"
OUTPUT_DIR = BASE_DIR / "data" / "output"


class PCBDetector:

    def __init__(
        self,
        confidence=0.05,
        image_size=640
    ):
        self.model = YOLO(
            str(MODEL_PATH)
        )

        self.confidence = confidence
        self.image_size = image_size

        print(
            "YOLO model loaded successfully."
        )

        print(
            f"Classes: "
            f"{len(self.model.names)}"
        )

    # =========================================================
    # DETECTION
    # =========================================================

    def detect(
        self,
        image_path,
        confidence=None
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        if confidence is None:

            confidence = (
                self.confidence
            )

        results = self.model(
            str(image_path),
            imgsz=self.image_size,
            conf=confidence,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is None:

            return detections

        # Original image dimensions
        image_width = (
            result.orig_shape[1]
        )

        image_height = (
            result.orig_shape[0]
        )

        # =====================================================
        # EXTRACT DETECTIONS
        # =====================================================

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence_score = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            # -------------------------------------------------
            # Center
            # -------------------------------------------------

            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2

            # -------------------------------------------------
            # Size
            # -------------------------------------------------

            width = (
                x2 - x1
            )

            height = (
                y2 - y1
            )

            # -------------------------------------------------
            # Normalized center
            # -------------------------------------------------

            normalized_center = {

                "x":
                    center_x
                    / image_width,

                "y":
                    center_y
                    / image_height
            }

            # -------------------------------------------------
            # Normalized size
            # -------------------------------------------------

            normalized_size = {

                "width":
                    width
                    / image_width,

                "height":
                    height
                    / image_height
            }

            # -------------------------------------------------
            # Detection object
            # -------------------------------------------------

            detection = {

                "class_id":
                    class_id,

                "class_name":
                    self.model.names[
                        class_id
                    ],

                "confidence":
                    confidence_score,

                "bbox": {

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },

                "center": {

                    "x": center_x,
                    "y": center_y
                },

                "normalized_center":
                    normalized_center,

                "size": {

                    "width": width,
                    "height": height
                },

                "normalized_size":
                    normalized_size
            }

            detections.append(
                detection
            )

        # =====================================================
        # SAVE DETECTION RESULT IMAGE
        # =====================================================

        try:

            annotated_result = (
                result.plot()
            )

            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            output_path = (
                OUTPUT_DIR
                / "detection_result.jpg"
            )

            import cv2

            cv2.imwrite(
                str(output_path),
                annotated_result
            )

            print(
                f"Result saved: "
                f"{output_path}"
            )

        except Exception as error:

            print(
                f"Could not save "
                f"detection image: {error}"
            )

        print(
            f"Valid detections: "
            f"{len(detections)}"
        )

        return detections


# =============================================================
# MODULE TEST
# =============================================================

if __name__ == "__main__":

    detector = PCBDetector()

    test_image = (
        BASE_DIR
        / "data"
        / "input"
        / "pcb_20261004_154241.jpg"
    )

    detections = detector.detect(
        test_image
    )

    print("\nDETECTIONS:")

    for detection in detections:

        print(
            detection
        )
