import sys
import uuid
import time
import importlib.util
from pathlib import Path


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
        detections
    ):

        print("\n[3/6] Running X-MCCV...")


        # ====================================================
        # CHECK ACTIVE TEMPLATE
        # ====================================================

        active_template = (
            self.template_manager
            .get_active_template()
        )


        # ====================================================
        # TEMPLATE MODE
        # ====================================================

        if active_template:

            print(
                f"Active template: "
                f"{active_template}"
            )

            template_data = (
                self.template_manager
                .load_template(
                    active_template
                )
            )

            expected_components = (
                template_data.get(
                    "components",
                    []
                )
            )

            print(
                f"Expected components: "
                f"{len(expected_components)}"
            )


        # ====================================================
        # TEMPORARY DEVELOPMENT MODE
        # ====================================================

        else:

            print(
                "No active template found."
            )

            print(
                "Using temporary detection "
                "self-reference mode."
            )

            expected_components = []

            for detection in detections:

                expected_components.append({

                    "class_id":
                        detection.get(
                            "class_id"
                        ),

                    "class_name":
                        detection.get(
                            "class_name"
                        ),

                    "count":
                        1,

                    "center":
                        detection.get(
                            "normalized_center"
                        ),

                    "orientation":
                        None,

                    "polarity":
                        None,

                    "size_shape":
                        detection.get(
                            "normalized_size"
                        ),

                    "spacing":
                        None
                })


        # ====================================================
        # RUN X-MCCV
        # ====================================================

        xmccv_results = self.xmccv.verify(
            None,
            detections,
            expected_components
        )


        # ====================================================
        # PRINT X-MCCV RESULTS
        # ====================================================

        print(
            "\n----------------------------------------"
        )

        print(
            "          X-MCCV RESULTS"
        )

        print(
            "----------------------------------------"
        )

        for result in xmccv_results:

            print(
                f"\nComponent: "
                f"{result.get('class_name')}"
            )

            reference_id = (
                result.get(
                    "reference_id"
                )
            )

            if reference_id:

                print(
                    f"Reference ID: "
                    f"{reference_id}"
                )

            print(
                f"Presence: "
                f"{result.get('presence')}"
            )

            print(
                f"Count: "
                f"{result.get('count')}"
            )

            print(
                f"Position: "
                f"{result.get('position')}"
            )

            print(
                f"Identity: "
                f"{result.get('identity')}"
            )

            print(
                f"Orientation: "
                f"{result.get('orientation')}"
            )

            print(
                f"Polarity: "
                f"{result.get('polarity')}"
            )

            print(
                f"Size/Shape: "
                f"{result.get('size_shape')}"
            )

            print(
                f"Spacing: "
                f"{result.get('spacing')}"
            )

            print(
                f"Component PASS: "
                f"{result.get('component_pass')}"
            )

            print(
                f"Reason: "
                f"{result.get('reason', 'N/A')}"
            )

        return xmccv_results


    # ========================================================
    # COMMON INSPECTION PIPELINE
    # ========================================================

    def inspect_image(
        self,
        image_path,
        inspection_id=None,
        use_hardware=True
    ):

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
                }
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
                    detections
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
                }
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
                }
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

                print(
                    "\nPASS -> Green LED "
                    "for 3 seconds"
                )

                self.hardware.pass_signal()

            else:

                print(
                    "\nFAIL -> Red LED + "
                    "Buzzer for 3 seconds"
                )

                self.hardware.fail_signal()

            time.sleep(3)

            self.hardware.reset()


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

    def inspect(self):

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
                    "Camera capture failed"
            }

        print(
            f"Captured image: "
            f"{image_path}"
        )

        self.hardware.image_captured()

        result = self.inspect_image(
            image_path=image_path,
            inspection_id=uuid.uuid4().hex[:12],
            use_hardware=True
        )

        result["input_type"] = "camera"

        return result


    # ==================================================
    # UPLOADED IMAGE INSPECTION
    # ==================================================

    def inspect_uploaded_image(
        self,
        image_path
    ):

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
                    "Image not found"
            }

        self.hardware.image_loaded()

        result = self.inspect_image(
            image_path=image_path,
            inspection_id=uuid.uuid4().hex[:12],
            use_hardware=False
        )

        result["input_type"] = "upload"

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
