import importlib.util
from pathlib import Path
from math import sqrt

from ultralytics import YOLO


BASE_DIR = Path("/home/pi4/PCBVISION")

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best.pt"
)

TEMPLATE_MANAGER_PATH = (
    BASE_DIR
    / "templates"
    / "templates.py"
)


# ============================================================
# LOAD TEMPLATE MANAGER
# ============================================================

spec = importlib.util.spec_from_file_location(
    "template_manager_module",
    TEMPLATE_MANAGER_PATH
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
# TEMPLATE BUILDER
# ============================================================

class TemplateBuilder:

    def __init__(
        self,
        confidence=0.25,
        position_tolerance=0.08
    ):

        self.model = YOLO(
            str(MODEL_PATH)
        )

        self.template_manager = (
            TemplateManager()
        )

        self.confidence = confidence
        self.position_tolerance = (
            position_tolerance
        )

        print(
            "Template Builder ready."
        )

        print(
            f"YOLO model: {MODEL_PATH}"
        )

        print(
            "Position correspondence "
            f"tolerance: {position_tolerance}"
        )

    # ========================================================
    # YOLO DETECTION
    # ========================================================

    def detect_reference_image(
        self,
        image_path
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Reference image not found: "
                f"{image_path}"
            )

        results = self.model(
            str(image_path),
            imgsz=640,
            conf=self.confidence,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is None:

            return detections

        image_width = (
            result.orig_shape[1]
        )

        image_height = (
            result.orig_shape[0]
        )

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2

            normalized_center = {

                "x":
                    center_x
                    / image_width,

                "y":
                    center_y
                    / image_height
            }

            normalized_size = {

                "width":
                    (x2 - x1)
                    / image_width,

                "height":
                    (y2 - y1)
                    / image_height
            }

            detections.append({

                "class_id":
                    class_id,

                "class_name":
                    self.model.names[
                        class_id
                    ],

                "confidence":
                    confidence,

                "normalized_center":
                    normalized_center,

                "normalized_size":
                    normalized_size
            })

        return detections

    # ========================================================
    # NORMALIZED POSITION DISTANCE
    # ========================================================

    def position_distance(
        self,
        center_a,
        center_b
    ):

        dx = (
            center_a["x"]
            - center_b["x"]
        )

        dy = (
            center_a["y"]
            - center_b["y"]
        )

        return sqrt(
            (dx * dx)
            + (dy * dy)
        )

    # ========================================================
    # FIND MATCHING COMPONENT
    # ========================================================

    def find_matching_component(
        self,
        detection,
        components
    ):

        best_component = None
        best_distance = None

        for component in components:

            if (
                component["class_name"]
                != detection["class_name"]
            ):
                continue

            distance = (
                self.position_distance(
                    detection[
                        "normalized_center"
                    ],
                    component[
                        "normalized_center"
                    ]
                )
            )

            if (
                distance
                > self.position_tolerance
            ):
                continue

            if (
                best_distance is None
                or distance < best_distance
            ):

                best_component = (
                    component
                )

                best_distance = (
                    distance
                )

        return (
            best_component,
            best_distance
        )

    # ========================================================
    # UPDATE COMPONENT
    # ========================================================

    def update_component(
        self,
        component,
        detection
    ):

        observation_count = (
            component[
                "observation_count"
            ]
        )

        old_center = component[
            "normalized_center"
        ]

        new_center = detection[
            "normalized_center"
        ]

        old_size = component[
            "normalized_size"
        ]

        new_size = detection[
            "normalized_size"
        ]

        # --------------------------------------------
        # Average normalized center
        # --------------------------------------------

        component[
            "normalized_center"
        ] = {

            "x":
                (
                    old_center["x"]
                    * observation_count
                    + new_center["x"]
                )
                / (
                    observation_count + 1
                ),

            "y":
                (
                    old_center["y"]
                    * observation_count
                    + new_center["y"]
                )
                / (
                    observation_count + 1
                )
        }

        # --------------------------------------------
        # Average normalized size
        # --------------------------------------------

        component[
            "normalized_size"
        ] = {

            "width":
                (
                    old_size["width"]
                    * observation_count
                    + new_size["width"]
                )
                / (
                    observation_count + 1
                ),

            "height":
                (
                    old_size["height"]
                    * observation_count
                    + new_size["height"]
                )
                / (
                    observation_count + 1
                )
        }

        component[
            "observation_count"
        ] += 1

        component[
            "confidence_sum"
        ] += detection[
            "confidence"
        ]

    # ========================================================
    # BUILD REFERENCE COMPONENTS
    # ========================================================

    def build_reference_components(
        self,
        reference_images
    ):

        components = []

        for image_index, image_path in enumerate(
            reference_images,
            start=1
        ):

            print()
            print(
                f"Processing reference "
                f"{image_index}/"
                f"{len(reference_images)}"
            )

            detections = (
                self.detect_reference_image(
                    image_path
                )
            )

            print(
                f"Detected components: "
                f"{len(detections)}"
            )

            for detection in detections:

                component, distance = (
                    self.find_matching_component(
                        detection,
                        components
                    )
                )

                if component is None:

                    component = {

                        "class_id":
                            detection[
                                "class_id"
                            ],

                        "class_name":
                            detection[
                                "class_name"
                            ],

                        "normalized_center":
                            {
                                "x":
                                    detection[
                                        "normalized_center"
                                    ]["x"],

                                "y":
                                    detection[
                                        "normalized_center"
                                    ]["y"]
                            },

                        "normalized_size":
                            {
                                "width":
                                    detection[
                                        "normalized_size"
                                    ]["width"],

                                "height":
                                    detection[
                                        "normalized_size"
                                    ]["height"]
                            },

                        "observation_count":
                            1,

                        "confidence_sum":
                            detection[
                                "confidence"
                            ]
                    }

                    components.append(
                        component
                    )

                else:

                    self.update_component(
                        component,
                        detection
                    )

        return components

    # ========================================================
    # FINALIZE TEMPLATE COMPONENTS
    # ========================================================

    def finalize_components(
        self,
        components,
        total_reference_images
    ):

        expected_components = []

        # Component must appear in at least
        # half of the reference images.

        minimum_observations = max(
            1,
            (
                total_reference_images
                + 1
            ) // 2
        )

        for component in components:

            if (
                component[
                    "observation_count"
                ]
                < minimum_observations
            ):
                continue

            average_confidence = (
                component[
                    "confidence_sum"
                ]
                /
                component[
                    "observation_count"
                ]
            )

            expected_components.append({

                "class_id":
                    component[
                        "class_id"
                    ],

                "class_name":
                    component[
                        "class_name"
                    ],

                "count":
                    1,

                "normalized_center":
                    {
                        "x":
                            component[
                                "normalized_center"
                            ]["x"],

                        "y":
                            component[
                                "normalized_center"
                            ]["y"]
                    },

                "orientation":
                    None,

                "polarity":
                    None,

                "size_shape":
                    {
                        "width":
                            component[
                                "normalized_size"
                            ]["width"],

                        "height":
                            component[
                                "normalized_size"
                            ]["height"]
                    },

                "spacing":
                    None,

                "reference_observations":
                    component[
                        "observation_count"
                    ],

                "average_confidence":
                    average_confidence
            })

        # --------------------------------------------
        # Sort consistently by position
        # --------------------------------------------

        expected_components.sort(

            key=lambda component: (

                component[
                    "normalized_center"
                ]["y"],

                component[
                    "normalized_center"
                ]["x"]
            )
        )

        # --------------------------------------------
        # Stable component IDs
        # --------------------------------------------

        for index, component in enumerate(
            expected_components,
            start=1
        ):

            component[
                "reference_id"
            ] = (
                f"C{index:03d}"
            )

        return expected_components

    # ========================================================
    # CREATE TEMPLATE
    # ========================================================

    def create_template(
        self,
        template_name,
        reference_images
    ):

        reference_images = [

            Path(image)

            for image in reference_images
        ]

        if len(reference_images) < 4:

            raise ValueError(
                "At least 4 reference images "
                "are required."
            )

        if len(reference_images) > 5:

            raise ValueError(
                "Use a maximum of 5 "
                "reference images."
            )

        for image in reference_images:

            if not image.exists():

                raise FileNotFoundError(
                    f"Reference image not found: "
                    f"{image}"
                )

        print()
        print(
            "================================"
        )
        print(
            "BUILDING PCB TEMPLATE"
        )
        print(
            "================================"
        )

        print(
            f"Template: {template_name}"
        )

        print(
            f"Reference images: "
            f"{len(reference_images)}"
        )

        # --------------------------------------------
        # Detect components from all references
        # --------------------------------------------

        raw_components = (
            self.build_reference_components(
                reference_images
            )
        )

        # --------------------------------------------
        # Create stable expected components
        # --------------------------------------------

        expected_components = (
            self.finalize_components(
                raw_components,
                len(reference_images)
            )
        )

        if not expected_components:

            raise RuntimeError(
                "No stable reference "
                "components found."
            )

        print()
        print(
            "Stable reference components:"
        )

        for component in (
            expected_components
        ):

            center = component[
                "normalized_center"
            ]

            print(

                f"{component['reference_id']} | "
                f"{component['class_name']} | "
                f"observed "
                f"{component['reference_observations']}/"
                f"{len(reference_images)} | "
                f"center=("
                f"{center['x']:.4f}, "
                f"{center['y']:.4f})"
            )

        # --------------------------------------------
        # Save template
        # --------------------------------------------

        template_data = (
            self.template_manager.create_template(

                template_name=
                    template_name,

                reference_images=
                    reference_images,

                expected_components=
                    expected_components
            )
        )

        print()
        print(
            "================================"
        )
        print(
            "TEMPLATE CREATED"
        )
        print(
            "================================"
        )

        print(
            f"Components: "
            f"{len(expected_components)}"
        )

        return template_data


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Template Builder module ready."
    )

    print(
        "Template builder uses normalized "
        "component coordinates."
    )	
