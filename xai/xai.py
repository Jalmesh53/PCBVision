import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


BASE_DIR = Path("/home/pi4/PCBVISION")
MODEL_PATH = BASE_DIR / "models" / "best.pt"
RESULTS_DIR = BASE_DIR / "results"


class XAIEngine:

    def __init__(self):

        self.model = YOLO(str(MODEL_PATH)).model
        self.detect_layer = self.model.model[23]

        self.features = []
        self.gradients = []
        self.hooks = []

        # Hook final classification convolution
        # of each YOLO detection scale.
        for branch in self.detect_layer.cv3:

            final_conv = branch[2]

            self.hooks.append(
                final_conv.register_forward_hook(
                    self._forward_hook
                )
            )

        print("[XAI] Model loaded.")
        print("[XAI] Grad-CAM hooks registered.")

    # ==================================================
    # FORWARD HOOK
    # ==================================================

    def _forward_hook(self, module, inputs, output):

        feature = inputs[0]

        self.features.append(feature)

        if feature.requires_grad:

            feature.register_hook(
                lambda grad: self.gradients.append(grad)
            )

    # ==================================================
    # GET CLASS ID
    # ==================================================

    def _get_class_id(self, detections):

        if not detections:
            return None

        detection = max(
            detections,
            key=lambda x: x.get("confidence", 0)
        )

        return int(detection["class_id"])

    # ==================================================
    # CREATE INSPECTION DIRECTORY
    # ==================================================

    def _get_inspection_dir(self, inspection_id):

        inspection_dir = (
            RESULTS_DIR /
            f"inspection_{inspection_id}"
        )

        inspection_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        return inspection_dir

    # ==================================================
    # EXISTING MODEL-DERIVED XAI
    # ==================================================

    def _generate_heatmap(
        self,
        image_path,
        detections,
        inspection_id
    ):

        start_time = time.perf_counter()

        self.features = []
        self.gradients = []

        class_id = self._get_class_id(detections)

        if class_id is None:

            raise RuntimeError(
                "No YOLO detection available for XAI."
            )

        print(
            f"[XAI] Target class: {class_id}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        original_h, original_w = image.shape[:2]

        # ----------------------------------------------
        # YOLO INPUT
        # ----------------------------------------------

        resized = cv2.resize(
            image,
            (640, 640)
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        tensor = torch.from_numpy(
            rgb
        ).permute(2, 0, 1).float() / 255.0

        tensor = tensor.unsqueeze(0)

        device = next(
            self.model.parameters()
        ).device

        tensor = tensor.to(device)

        tensor.requires_grad_(True)

        self.model.eval()
        self.model.zero_grad(
            set_to_none=True
        )

        self.features = []
        self.gradients = []

        # ----------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------

        with torch.enable_grad():

            _ = self.model(tensor)

            if len(self.features) != 3:

                raise RuntimeError(
                    f"Expected 3 XAI feature maps, "
                    f"got {len(self.features)}"
                )

            print(
                "[XAI] Feature maps:",
                [
                    tuple(feature.shape)
                    for feature in self.features
                ]
            )

            # ------------------------------------------
            # DIFFERENTIABLE CLASS RESPONSE
            # ------------------------------------------

            class_responses = []

            for i, branch in enumerate(
                self.detect_layer.cv3
            ):

                final_conv = branch[2]

                feature = self.features[i]

                weight = final_conv.weight[
                    class_id
                ]

                bias = final_conv.bias[
                    class_id
                ]

                response = (
                    feature *
                    weight.view(
                        1,
                        -1,
                        1,
                        1
                    )
                ).sum(
                    dim=1
                ) + bias

                class_responses.append(
                    response
                )

                print(
                    f"[XAI] Scale {i}:",
                    tuple(response.shape),
                    "requires_grad:",
                    response.requires_grad
                )

            # Combine scales
            target = 0.0

            for response in class_responses:

                target = (
                    target +
                    response.mean()
                )

            print(
                "[XAI] Target requires grad:",
                target.requires_grad
            )

            print(
                "[XAI] Target grad_fn:",
                type(target.grad_fn).__name__
                if target.grad_fn
                else None
            )

            if not target.requires_grad:

                raise RuntimeError(
                    "XAI target is not connected "
                    "to computation graph."
                )

            target.backward()

        # ----------------------------------------------
        # GRADIENT VALIDATION
        # ----------------------------------------------

        if len(self.gradients) == 0:

            raise RuntimeError(
                "No gradients captured for XAI."
            )

        print(
            "[XAI] Gradients captured:",
            len(self.gradients)
        )

        cams = []

        gradients_by_size = {
            tuple(g.shape[-2:]): g
            for g in self.gradients
        }

        # ----------------------------------------------
        # LAYERCAM
        # ----------------------------------------------

        for feature in self.features:

            size = tuple(
                feature.shape[-2:]
            )

            grad = gradients_by_size.get(
                size
            )

            if grad is None:
                continue

            # Positive gradient × activation
            cam = (
                torch.relu(grad) *
                feature
            )

            cam = cam.sum(
                dim=1
            )

            cam = torch.relu(
                cam
            )

            cam = (
                cam.squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            if cam.max() > cam.min():

                cam = (
                    cam - cam.min()
                ) / (
                    cam.max() - cam.min()
                )

            else:

                cam = np.zeros_like(
                    cam
                )

            cam = cv2.resize(
                cam,
                (640, 640),
                interpolation=cv2.INTER_LINEAR
            )

            cams.append(cam)

        if not cams:

            raise RuntimeError(
                "Unable to generate XAI activation map."
            )

        # Combine multi-scale maps
        heatmap = np.maximum.reduce(
            cams
        )

        heatmap = np.clip(
            heatmap * 255,
            0,
            255
        ).astype(np.uint8)

        # Resize to original image
        heatmap = cv2.resize(
            heatmap,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Smooth
        heatmap = cv2.GaussianBlur(
            heatmap,
            (0, 0),
            3
        )

        # ----------------------------------------------
        # OUTPUT PATHS
        # ----------------------------------------------

        inspection_dir = (
            self._get_inspection_dir(
                inspection_id
            )
        )

        heatmap_path = (
            inspection_dir /
            "heatmap.jpg"
        )

        overlay_path = (
            inspection_dir /
            "xai_overlay.jpg"
        )

        # ----------------------------------------------
        # COLORED HEATMAP
        # ----------------------------------------------

        colored_heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        cv2.imwrite(
            str(heatmap_path),
            colored_heatmap
        )

        # ----------------------------------------------
        # OVERLAY
        # ----------------------------------------------

        overlay = cv2.addWeighted(
            image,
            0.55,
            colored_heatmap,
            0.45,
            0
        )

        class_name = self.model.names[
            class_id
        ]

        cv2.putText(
            overlay,
            f"XAI: {class_name}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imwrite(
            str(overlay_path),
            overlay
        )

        processing_time = (
            time.perf_counter()
            - start_time
        )

        print(
            f"[XAI] Heatmap saved: "
            f"{heatmap_path}"
        )

        print(
            f"[XAI] Overlay saved: "
            f"{overlay_path}"
        )

        print(
            f"[XAI] Processing time: "
            f"{processing_time:.3f}s"
        )

        return {

            "status": "success",

            "mode": "model_derived",

            "class_id": class_id,

            "class_name": class_name,

            "heatmap_path":
                str(heatmap_path),

            "overlay_path":
                str(overlay_path),

            "processing_time":
                processing_time
        }

    # ==================================================
    # MISSING COMPONENT EXPLANATION
    # ==================================================

    def _generate_missing_component_explanation(
        self,
        image_path,
        x_mccv_results,
        inspection_id
    ):

        start_time = time.perf_counter()

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        original_h, original_w = (
            image.shape[:2]
        )

        # ----------------------------------------------
        # FIND MISSING COMPONENTS
        # ----------------------------------------------

        missing_components = []

        for result in x_mccv_results:

            if (
                not result.get("presence", True)
                and result.get("reason")
                == "Component missing"
            ):

                missing_components.append(
                    result
                )

        if not missing_components:

            raise RuntimeError(
                "No missing component found "
                "in X-MCCV results."
            )

        # ----------------------------------------------
        # CREATE VISUAL EVIDENCE
        # ----------------------------------------------

        overlay = image.copy()

        for result in missing_components:

            reference_id = result.get(
                "reference_id",
                "UNKNOWN"
            )

            class_name = result.get(
                "class_name",
                "Unknown"
            )

            expected_center = result.get(
                "expected_normalized_center"
            )

            # ------------------------------------------
            # EXPECTED LOCATION
            # ------------------------------------------

            if (
                expected_center is not None
                and
                "x" in expected_center
                and
                "y" in expected_center
            ):

                center_x = int(
                    expected_center["x"]
                    * original_w
                )

                center_y = int(
                    expected_center["y"]
                    * original_h
                )

                # Size is intentionally moderate
                # because there is no actual bounding
                # box when the component is missing.
                box_width = int(
                    original_w * 0.12
                )

                box_height = int(
                    original_h * 0.12
                )

                x1 = max(
                    0,
                    center_x - box_width // 2
                )

                y1 = max(
                    0,
                    center_y - box_height // 2
                )

                x2 = min(
                    original_w - 1,
                    center_x + box_width // 2
                )

                y2 = min(
                    original_h - 1,
                    center_y + box_height // 2
                )

                # Expected location rectangle
                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    4
                )

                # Center marker
                cv2.drawMarker(
                    overlay,
                    (center_x, center_y),
                    (0, 255, 255),
                    cv2.MARKER_CROSS,
                    40,
                    3
                )

                # Label above expected region
                label = (
                    f"MISSING: "
                    f"{class_name} ({reference_id})"
                )

                label_y = max(
                    35,
                    y1 - 12
                )

                cv2.putText(
                    overlay,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    overlay,
                    "EXPECTED LOCATION",
                    (x1, min(
                        original_h - 15,
                        y2 + 30
                    )),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                print(
                    f"[XAI] Expected location for "
                    f"{class_name} ({reference_id}): "
                    f"({expected_center['x']:.4f}, "
                    f"{expected_center['y']:.4f})"
                )

            else:

                # Fallback if normalized location
                # is unavailable.
                cv2.putText(
                    overlay,
                    f"MISSING: {class_name}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    overlay,
                    f"Reference: {reference_id}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # ----------------------------------------------
        # INFORMATION HEADER
        # ----------------------------------------------

        cv2.rectangle(
            overlay,
            (0, 0),
            (original_w, 45),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            overlay,
            "XAI: TEMPLATE-BASED MISSING COMPONENT EVIDENCE",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------
        # SAVE OUTPUTS
        # ----------------------------------------------

        inspection_dir = (
            self._get_inspection_dir(
                inspection_id
            )
        )

        evidence_path = (
            inspection_dir /
            "missing_component.jpg"
        )

        overlay_path = (
            inspection_dir /
            "xai_overlay.jpg"
        )

        # Original image as evidence
        cv2.imwrite(
            str(evidence_path),
            image
        )

        # Annotated missing-component image
        cv2.imwrite(
            str(overlay_path),
            overlay
        )

        processing_time = (
            time.perf_counter()
            - start_time
        )

        print(
            "[XAI] Missing component detected."
        )

        print(
            f"[XAI] Missing components: "
            f"{len(missing_components)}"
        )

        print(
            f"[XAI] Evidence saved: "
            f"{evidence_path}"
        )

        print(
            f"[XAI] Overlay saved: "
            f"{overlay_path}"
        )

        print(
            f"[XAI] Processing time: "
            f"{processing_time:.3f}s"
        )

        return {

            "status": "success",

            "mode": "missing_component",

            "missing_components":
                missing_components,

            "evidence_path":
                str(evidence_path),

            "overlay_path":
                str(overlay_path),

            "heatmap_path":
                None,

            "processing_time":
                processing_time
        }

    # ==================================================
    # PUBLIC EXPLAIN METHOD
    # ==================================================

    def explain(
        self,
        image_path,
        detections,
        inspection_id="test",
        x_mccv_results=None
    ):

        try:

            # ------------------------------------------
            # NORMAL YOLO XAI
            # ------------------------------------------

            if detections:

                return self._generate_heatmap(
                    image_path,
                    detections,
                    inspection_id
                )

            # ------------------------------------------
            # MISSING COMPONENT XAI
            # ------------------------------------------

            if x_mccv_results:

                return (
                    self
                    ._generate_missing_component_explanation(
                        image_path,
                        x_mccv_results,
                        inspection_id
                    )
                )

            raise RuntimeError(
                "No YOLO detection or X-MCCV "
                "failure information available."
            )

        except Exception as e:

            print(
                f"[XAI] FAILED: {e}"
            )

            return {
                "status": "failed",
                "error": str(e)
            }

    # ==================================================
    # CLOSE
    # ==================================================

    def close(self):

        for hook in self.hooks:

            hook.remove()

        self.hooks.clear()


# ======================================================
# STANDALONE TEST
# ======================================================

if __name__ == "__main__":

    image_path = (
        BASE_DIR /
        "data" /
        "input" /
        "test.jpg"
    )

    print(
        "[XAI] Starting standalone test..."
    )

    detector = YOLO(
        str(MODEL_PATH)
    )

    results = detector(
        str(image_path),
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            detections.append({

                "class_id":
                    class_id,

                "class_name":
                    detector.names[class_id],

                "confidence":
                    float(box.conf[0])
            })

    print(
        f"[XAI] Detections: "
        f"{len(detections)}"
    )

    xai = XAIEngine()

    output = xai.explain(
        image_path,
        detections,
        inspection_id="test"
    )

    print(
        "\nXAI RESULT:"
    )

    print(output)

    xai.close()
