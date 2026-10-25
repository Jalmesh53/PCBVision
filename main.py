import sys
import uuid
import time
import importlib.util
from pathlib import Path


# YOLO classes that represent known PCB defects. These names must match
# the classes in models/best.pt. A detection flags the board for FAIL; it
# is not a claim that the model detects every possible defect type.
PCB_DEFECT_CLASSES = {
    "missing_pad",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
}


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path("/home/pi4/PCBVISION")

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# MODULES
# ============================================================

from camera.camera import capture_image
from detection.detector import PCBDetector
from ocr.ocr import ComponentOCR
from verification.x_mccv import XMCCV
from decision.decision import DecisionEngine
from hardware.hardware import HardwareController
from xai.xai import XAIEngine


# ============================================================
# LOAD TEMPLATE MANAGER
# ============================================================

TEMPLATE_MANAGER_PATH = (
    BASE_DIR /
    "templates" /
    "templates.py"
)

spec = importlib.util.spec_from_file_location(
    "template_manager_module",
    TEMPLATE_MANAGER_PATH
)

if spec is None or spec.loader is None:
    raise ImportError(
        "Unable to load templates.py"
    )

template_manager_module = (
    importlib.util.module_from_spec(spec)
)

spec.loader.exec_module(
    template_manager_module
)

TemplateManager = (
    template_manager_module.TemplateManager
)


# ============================================================
# PCBVISION
# ============================================================

class PCBVision:

    def __init__(self):

        print("\n========================================")
        print("        PCBVISION INITIALIZING")
        print("========================================")

        self.detector = PCBDetector()

        self.ocr = ComponentOCR()

        self.xmccv = XMCCV(
            position_tolerance=0.10,
            orientation_tolerance=20.0,
            size_tolerance=0.20,
            spacing_tolerance=0.20
        )

        self.decision_engine = DecisionEngine()

        self.hardware = HardwareController()

        self.xai = XAIEngine()

        self.template_manager = TemplateManager()

        print("\nPCBVISION READY.")


    # ========================================================
    # OCR
    # ========================================================

    def run_ocr(
        self,
        image_path,
        detections
    ):

        print("\n[2/6] Running OCR...")

        import cv2

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "Unable to read image for OCR."
            )

            return detections

        output_dir = (
            BASE_DIR /
            "data" /
            "output"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for index, detection in enumerate(
            detections,
            start=1
        ):

            bbox = detection.get(
                "bbox"
            )

            if not bbox:

                detection["ocr_text"] = ""

                continue

            try:

                x1 = max(
                    0,
                    int(bbox["x1"])
                )

                y1 = max(
                    0,
                    int(bbox["y1"])
                )

                x2 = min(
                    image.shape[1],
                    int(bbox["x2"])
                )

                y2 = min(
                    image.shape[0],
                    int(bbox["y2"])
                )

                crop = image[
                    y1:y2,
                    x1:x2
                ]

                if crop.size == 0:

                    detection["ocr_text"] = ""

                    continue

                crop_path = (
                    output_dir /
                    f"ocr_component_{index}.jpg"
                )

                cv2.imwrite(
                    str(crop_path),
                    crop
                )

                text = self.ocr.extract_text(
                    crop_path
                )

                detection["ocr_text"] = text

                print(
                    f"Component {index}: "
                    f"{detection.get('class_name')} "
                    f"| OCR: "
                    f"{text if text else '[none]'}"
                )

            except Exception as error:

                detection["ocr_text"] = ""

                print(
                    f"OCR error for component "
                    f"{index}: {error}"
                )

        return detections


    # ========================================================
    # X-MCCV
    # ========================================================

    def run_xmccv(
        self,
        detections,
        use_template=False,
        template_name=None
    ):

        print("\n[3/6] Running X-MCCV...")

        # Template verification is opt-in. Prefer the template explicitly
        # selected by the user; use the active template only for legacy
        # callers that enable template mode without supplying a name.
        active_template = None
        if use_template:
            active_template = (
                template_name
                if template_name
                else self.template_manager.get_active_template()
            )

            if not active_template:
                raise ValueError(
                    "Template inspection was requested, but no template was selected."
                )

            available_templates = self.template_manager.list_templates()
            if active_template not in available_templates:
                raise ValueError(
                    f"Template not found: {active_template}"
                )

        if active_template:
            print(f"Selected template: {active_template}")

            template_data = (
                self.template_manager.load_template(active_template)
            )

            expected_components = template_data.get("components", [])

            print(f"Expected components: {len(expected_components)}")

            xmccv_results = self.xmccv.verify(
                None,
                detections,
                expected_components
            )

        else:
            # Detection-only mode must NOT pass detections through the
            # template verifier. There are no reference criteria here.
            print("General inspection: detection and measurement mode.")

            xmccv_results = []
            class_counts = {}

            for detection in detections:
                class_name = detection.get("class_name", "Unknown")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            for detection in detections:
                class_name = detection.get("class_name", "Unknown")
                bbox = detection.get("bbox") or {}
                center = detection.get("normalized_center") or {}
                size = detection.get("normalized_size") or {}

                width = size.get("width")
                height = size.get("height")

                bbox_width = None
                bbox_height = None
                if all(key in bbox for key in ("x1", "x2", "y1", "y2")):
                    bbox_width = bbox["x2"] - bbox["x1"]
                    bbox_height = bbox["y2"] - bbox["y1"]

                # Nearest detected component, measured in normalized
                # image coordinates. This is not a physical distance.
                spacing = None
                nearest_distance = None
                nearest_component = None

                cx = center.get("x")
                cy = center.get("y")

                if cx is not None and cy is not None:
                    for other in detections:
                        if other is detection:
                            continue

                        other_center = other.get("normalized_center") or {}
                        ox = other_center.get("x")
                        oy = other_center.get("y")

                        if ox is None or oy is None:
                            continue

                        distance = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5

                        if nearest_distance is None or distance < nearest_distance:
                            nearest_distance = distance
                            nearest_component = other.get("class_name", "Unknown")

                    if nearest_distance is not None:
                        spacing = {
                            "nearest_component": nearest_component,
                            "normalized_center_distance": round(nearest_distance, 4),
                            "unit": "normalized_image_coordinates"
                        }

                xmccv_results.append({
                    "class_name": class_name,
                    "class_id": detection.get("class_id"),
                    "presence": True,
                    "count": class_counts[class_name],
                    "position": {
                        "x": cx,
                        "y": cy
                    },
                    "identity": class_name,
                    "confidence": detection.get("confidence"),
                    "orientation": None,
                    "polarity": None,
                    "size_shape": {
                        "width_normalized": width,
                        "height_normalized": height,
                        "aspect_ratio": (
                            round(width / height, 4)
                            if isinstance(width, (int, float))
                            and isinstance(height, (int, float))
                            and height > 0
                            else None
                        ),
                        "bbox_pixels": {
                            "width": bbox_width,
                            "height": bbox_height
                        }
                    },
                    "normalized_size": size,
                    "spacing": spacing,
                    "component_pass": None,
                    "verification_mode": "detection_only",
                    "unexpected": False,
                    "reason": None
                })

        # Print X-MCCV results.
        print("\n----------------------------------------")
        print("          X-MCCV RESULTS")
        print("----------------------------------------")

        for result in xmccv_results:
            print(f"\nComponent: {result.get('class_name')}")
            print(f"Presence: {result.get('presence')}")
            print(f"Count: {result.get('count')}")
            print(f"Position: {result.get('position')}")
            print(f"Identity: {result.get('identity')}")
            print(f"Orientation: {result.get('orientation')}")
            print(f"Polarity: {result.get('polarity')}")
            print(f"Size/Shape: {result.get('size_shape')}")
            print(f"Spacing: {result.get('spacing')}")
            component_pass = result.get("component_pass")

            if component_pass is True:
                component_status = "PASS"
            elif component_pass is False:
                component_status = "FAIL"
            else:
                component_status = result.get("component_status") or "UNVERIFIED"

            print(f"Component Status: {component_status}")
            print(f"Component PASS: {result.get('component_pass')}")
            print(f"Reason: {result.get('reason', 'N/A')}")

        return xmccv_results


    # ========================================================
    # COMMON INSPECTION PIPELINE
    # ========================================================

    def inspect_image(
        self,
        image_path,
        inspection_id=None,
        use_hardware=True,
        inspection_start=None,
        use_template=False,
        template_name=None
    ):

        if inspection_start is None:
            inspection_start = time.perf_counter()

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Inspection image not found: "
                f"{image_path}"
            )

        if inspection_id is None:

            inspection_id = (
                uuid.uuid4().hex[:12]
            )


        print("\n================================")

        print(
            f"INSPECTION ID: "
            f"{inspection_id}"
        )

        print(
            "================================"
        )

        print(
            f"\nImage: {image_path}"
        )


        # ==================================================
        # 1. YOLO
        # ==================================================

        if use_hardware:

            self.hardware.validating_model()

        print(
            "\n[1/6] Running YOLO detection..."
        )

        try:

            detections = self.detector.detect(
                image_path
            )

        except Exception as error:

            print(
                f"Detection failed: {error}"
            )

            if use_hardware:

                self.hardware.error(
                    "YOLO ERROR"
                )

                time.sleep(3)

                self.hardware.reset()

            return {
                "status": "FAIL",
                "inspection_id":
                    inspection_id,
                "input_type": "image",
                "image_path":
                    str(image_path),
                "detection_image":
                    None,
                "detections": [],
                "xmccv": [],
                "decision": {
                    "overall_status":
                        "FAIL",
                    "reason":
                        "YOLO detection failed",
                    "passed_components":
                        0,
                    "failed_components":
                        0
                },
                "xai": {
                    "status":
                        "error",
                    "error":
                        str(error)
                },
                "inspection_time":
                    round(
                        time.perf_counter()
                        - inspection_start,
                        2
                    )
            }


        print(
            f"Detected components: "
            f"{len(detections)}"
        )

        for detection in detections:

            print(
                f"  "
                f"{detection['class_name']} "
                f"| "
                f"{detection['confidence']:.2f}"
            )


        # ==================================================
        # 2. OCR
        # ==================================================

        if use_hardware:

            self.hardware.reading_parts()

        detections = self.run_ocr(
            image_path,
            detections
        )


        # ==================================================
        # 3. X-MCCV
        # ==================================================

        if use_hardware:

            self.hardware.applying_xmccv()

        try:

            xmccv_results = (
                self.run_xmccv(
                    detections,
                    use_template=use_template,
                    template_name=template_name
                )
            )

        except Exception as error:

            print(
                f"X-MCCV failed: {error}"
            )

            if use_hardware:

                self.hardware.error(
                    "X-MCCV ERROR"
                )

                time.sleep(3)

                self.hardware.reset()

            return {
                "status": "FAIL",
                "inspection_id":
                    inspection_id,
                "input_type": "image",
                "image_path":
                    str(image_path),
                "detection_image":
                    None,
                "detections":
                    detections,
                "xmccv": [],
                "decision": {
                    "overall_status":
                        "FAIL",
                    "reason":
                        "X-MCCV verification failed",
                    "passed_components":
                        0,
                    "failed_components":
                        0
                },
                "xai": {
                    "status":
                        "error",
                    "error":
                        str(error)
                },
                "inspection_time":
                    round(
                        time.perf_counter()
                        - inspection_start,
                        2
                    )
            }


        # ==================================================
        # 4. DECISION
        # ==================================================

        if use_hardware:

            self.hardware.making_decision()

        print(
            "\n[4/6] Running decision engine..."
        )

        try:

            decision = (
                self.decision_engine.evaluate(
                    xmccv_results
                )
            )

            # Inspect the existing YOLO detections for known PCB defect classes.
            # Keep component/X-MCCV processing unchanged; a recognized defect
            # detection overrides an otherwise UNVERIFIED/PASS decision to FAIL.
            detected_defects = [
                {
                    "class_name": detection.get("class_name"),
                    "confidence": detection.get("confidence"),
                    "bbox": detection.get("bbox"),
                }
                for detection in detections
                if detection.get("class_name") in PCB_DEFECT_CLASSES
            ]

            if detected_defects:
                defect_summary = ", ".join(
                    f"{item['class_name']} "
                    f"({float(item['confidence']):.2f})"
                    if isinstance(item.get("confidence"), (int, float))
                    else str(item["class_name"])
                    for item in detected_defects
                )
                decision["overall_status"] = "FAIL"
                decision["reason"] = (
                    "Potential PCB defect detected by YOLO: "
                    f"{defect_summary}"
                )

            decision["detected_defects"] = detected_defects

            print("\n----------------------------------------")
            print("          X-MCCV RESULTS")
            print("----------------------------------------")

            for result in xmccv_results:
                print(f"\nComponent: {result.get('class_name')}")
                print(f"Presence: {result.get('presence')}")
                print(f"Count: {result.get('count')}")
                print(f"Position: {result.get('position')}")
                print(f"Identity: {result.get('identity')}")
                print(f"Orientation: {result.get('orientation')}")
                print(f"Polarity: {result.get('polarity')}")
                print(f"Size/Shape: {result.get('size_shape')}")
                print(f"Spacing: {result.get('spacing')}")
                print(f"Component Status: {result.get('component_status', 'UNVERIFIED')}")
                print(f"Component PASS: {result.get('component_pass')}")
                print(f"Reason: {result.get('reason', 'N/A')}")

    

            # Preserve the decision engine result for both general and template inspections.

        except Exception as error:

            print(
                f"Decision engine failed: "
                f"{error}"
            )

            if use_hardware:

                self.hardware.error(
                    "DECISION ERROR"
                )

                time.sleep(3)

                self.hardware.reset()

            return {
                "status": "FAIL",
                "inspection_id":
                    inspection_id,
                "input_type": "image",
                "image_path":
                    str(image_path),
                "detection_image":
                    None,
                "detections":
                    detections,
                "xmccv":
                    xmccv_results,
                "decision": {
                    "overall_status":
                        "FAIL",
                    "reason":
                        "Decision engine failed",
                    "passed_components":
                        0,
                    "failed_components":
                        0
                },
                "xai": {
                    "status":
                        "error",
                    "error":
                        str(error)
                },
                "inspection_time":
                    round(
                        time.perf_counter()
                        - inspection_start,
                        2
                    )
            }


        print(
            f"\nSTATUS: "
            f"{decision['overall_status']}"
        )

        print(
            f"Passed components: "
            f"{decision['passed_components']}"
        )

        print(
            f"Failed components: "
            f"{decision['failed_components']}"
        )

        print(
            f"Reason: "
            f"{decision.get('reason', 'N/A')}"
        )


        # ==================================================
        # 5. XAI
        #
        # XAI runs for BOTH PASS and FAIL.
        # ==================================================

        if use_hardware:

            self.hardware.generating_xai()

        print(
            "\n[5/6] Generating XAI explanation..."
        )

        xai_result = {}

        try:

            # IMPORTANT:
            # Pass X-MCCV results to XAI.
            #
            # If detections exist:
            #     XAI uses model-derived LayerCAM.
            #
            # If no detections exist but X-MCCV
            # reports a missing component:
            #     XAI uses the missing-component
            #     explanation path.

            xai_result = self.xai.explain(
                image_path=image_path,
                detections=detections,
                inspection_id=inspection_id,
                x_mccv_results=xmccv_results
            )

            print(
                f"XAI status: "
                f"{xai_result.get('status')}"
            )

            if (
                xai_result.get(
                    "status"
                )
                == "success"
            ):

                print(
                    f"Heatmap: "
                    f"{xai_result.get('heatmap_path')}"
                )

                print(
                    f"Overlay: "
                    f"{xai_result.get('overlay_path')}"
                )

                processing_time = (
                    xai_result.get(
                        "processing_time"
                    )
                )

                if processing_time is not None:

                    print(
                        f"XAI time: "
                        f"{processing_time:.3f}s"
                    )

                xai_mode = (
                    xai_result.get(
                        "mode"
                    )
                )

                if xai_mode:

                    print(
                        f"XAI mode: "
                        f"{xai_mode}"
                    )

            else:

                print(
                    f"XAI error: "
                    f"{xai_result.get('error')}"
                )

        except Exception as error:

            print(
                f"XAI failed: {error}"
            )

            xai_result = {
                "status":
                    "error",
                "error":
                    str(error)
            }


        # ==================================================
        # 6. FINAL RESULT
        # ==================================================

        if use_hardware:

            self.hardware.inspection_complete()

        print(
            "\n[6/6] Preparing inspection result..."
        )

        detection_image = (
            BASE_DIR /
            "data" /
            "output" /
            "detection_result.jpg"
        )


        result = {

            "status":
                decision[
                    "overall_status"
                ],

            "inspection_id":
                inspection_id,

            "input_type":
                "image",

            "image_path":
                str(image_path),

            "detection_image":
                str(detection_image),

            "detections":
                detections,

            "detected_defects":
                decision.get("detected_defects", []),

            "xmccv":
                xmccv_results,

            "decision":
                decision,

            "xai":
                xai_result
        }


        # ==================================================
        # COMPLETE RESULT
        # ==================================================

        print(
            "\n========================================"
        )

        print(
            "          INSPECTION RESULT"
        )

        print(
            "========================================"
        )

        print(
            f"STATUS: "
            f"{result['status']}"
        )

        print(
            f"Reason: "
            f"{decision.get('reason', 'N/A')}"
        )

        print(
            f"Passed components: "
            f"{decision.get('passed_components', 0)}"
        )

        print(
            f"Failed components: "
            f"{decision.get('failed_components', 0)}"
        )

        print(
            f"Detection image: "
            f"{detection_image}"
        )

        if (
            xai_result.get(
                "status"
            )
            == "success"
        ):

            print(
                f"XAI heatmap: "
                f"{xai_result.get('heatmap_path')}"
            )

            print(
                f"XAI overlay: "
                f"{xai_result.get('overlay_path')}"
            )

        else:

            print(
                "XAI explanation: "
                "Not available"
            )

        print(
            "========================================"
        )


        # ==================================================
        # FINAL HARDWARE INDICATION
        #
        # Hold result for 3 seconds.
        # ==================================================

        if use_hardware:

            if result["status"] == "PASS":

                print("\nPASS -> Green LED for 3 seconds")
                self.hardware.pass_signal()
                time.sleep(3)

            elif result["status"] == "FAIL":

                print("\nFAIL -> Red LED + Buzzer for 3 seconds")
                self.hardware.fail_signal()
                time.sleep(3)

            else:

                print(
                    "\nUNVERIFIED -> No PASS/FAIL hardware signal"
                )

            self.hardware.reset()


        inspection_time = round(
            time.perf_counter()
            - inspection_start,
            2
        )

        result["inspection_time"] = inspection_time

        print(
            f"\nTotal inspection time: "
            f"{inspection_time:.2f}s"
        )


        print(
            "\n================================"
        )

        print(
            "       INSPECTION COMPLETE"
        )

        print(
            "================================"
        )

        return result


    # ==================================================
    # CAMERA INSPECTION
    # ==================================================

    def inspect(self, template_name=None):

        if template_name is not None:
            template_name = str(template_name).strip()
            if not template_name:
                raise ValueError("Template name cannot be empty.")

            if template_name not in self.template_manager.list_templates():
                raise ValueError(f"Template not found: {template_name}")

        inspection_start = time.perf_counter()

        print(
            "\n[CAMERA MODE]"
        )

        self.hardware.capturing_image()

        print(
            "\nCapturing PCB image..."
        )

        image_path = capture_image()

        if image_path is None:

            self.hardware.error(
                "CAMERA ERROR"
            )

            time.sleep(3)

            self.hardware.reset()

            return {
                "status": "FAIL",
                "inspection_id": None,
                "input_type": "camera",
                "reason":
                    "Camera capture failed",
                "inspection_time":
                    round(
                        time.perf_counter()
                        - inspection_start,
                        2
                    )
            }

        print(
            f"Captured image: "
            f"{image_path}"
        )

        self.hardware.image_captured()

        result = self.inspect_image(
            image_path=image_path,
            inspection_id=uuid.uuid4().hex[:12],
            use_hardware=True,
            inspection_start=inspection_start,
            use_template=template_name is not None,
            template_name=template_name
        )

        result["input_type"] = "camera"
        if template_name is not None:
            result["template_name"] = template_name

        return result


    # ==================================================
    # UPLOADED IMAGE INSPECTION
    # ==================================================

    def inspect_uploaded_image(
        self,
        image_path,
        template_name=None
    ):

        if template_name is not None:
            template_name = str(template_name).strip()
            if not template_name:
                raise ValueError("Template name cannot be empty.")

            if template_name not in self.template_manager.list_templates():
                raise ValueError(f"Template not found: {template_name}")

        inspection_start = time.perf_counter()

        print(
            "\n[UPLOADED IMAGE MODE]"
        )

        self.hardware.loading_image()

        print(
            f"\nLoading image: "
            f"{image_path}"
        )

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            self.hardware.error(
                "IMAGE ERROR"
            )

            time.sleep(3)

            self.hardware.reset()

            return {
                "status": "FAIL",
                "inspection_id": None,
                "input_type": "upload",
                "reason":
                    "Image not found",
                "inspection_time":
                    round(
                        time.perf_counter()
                        - inspection_start,
                        2
                    )
            }

        self.hardware.image_loaded()

        result = self.inspect_image(
            image_path=image_path,
            inspection_id=uuid.uuid4().hex[:12],
            use_hardware=True,
            inspection_start=inspection_start,
            use_template=template_name is not None,
            template_name=template_name
        )

        result["input_type"] = "upload"
        if template_name is not None:
            result["template_name"] = template_name

        return result


    # ==================================================
    # SHUTDOWN
    # ==================================================

    def shutdown(self):

        try:

            self.xai.close()

        except Exception:

            pass

        try:

            self.hardware.reset()

        except Exception:

            pass


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    system = None

    try:

        system = PCBVision()

        result = system.inspect()

        print(
            "\nRESULT SUMMARY:"
        )

        print(
            f"Status: "
            f"{result.get('status')}"
        )

        print(
            f"Inspection ID: "
            f"{result.get('inspection_id')}"
        )

        print(
            f"Input type: "
            f"{result.get('input_type')}"
        )

        print(
            f"Inspection Time: "
            f"{result.get('inspection_time', 'N/A')}s"
        )

        print(
            f"Image: "
            f"{result.get('image_path')}"
        )

        print(
            f"Detection image: "
            f"{result.get('detection_image')}"
        )

        decision = result.get(
            "decision",
            {}
        )

        print(
            f"Reason: "
            f"{decision.get('reason', 'N/A')}"
        )

        print(
            f"Passed components: "
            f"{decision.get('passed_components', 0)}"
        )

        print(
            f"Failed components: "
            f"{decision.get('failed_components', 0)}"
        )

        xai = result.get(
            "xai",
            {}
        )

        if xai.get(
            "status"
        ) == "success":

            print(
                f"XAI heatmap: "
                f"{xai.get('heatmap_path')}"
            )

            print(
                f"XAI overlay: "
                f"{xai.get('overlay_path')}"
            )

            print(
                f"XAI mode: "
                f"{xai.get('mode', 'N/A')}"
            )

        else:

            print(
                f"XAI status: "
                f"{xai.get('status', 'N/A')}"
            )


    except KeyboardInterrupt:

        print(
            "\nInspection interrupted."
        )

    except Exception as error:

        print(
            f"\nPCBVISION ERROR: "
            f"{error}"
        )

    finally:

        if system is not None:

            system.shutdown()
