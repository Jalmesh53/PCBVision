import sys
import importlib.util
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path("/home/pi4/PCBVISION")

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# CAMERA
# ============================================================

from camera.camera import capture_image


# ============================================================
# LOAD TEMPLATE BUILDER
# ============================================================

TEMPLATE_BUILDER_PATH = (
    BASE_DIR /
    "templates" /
    "template_builder.py"
)

spec = importlib.util.spec_from_file_location(
    "template_builder_module",
    TEMPLATE_BUILDER_PATH
)

if spec is None or spec.loader is None:
    raise ImportError(
        "Unable to load template_builder.py"
    )

template_builder_module = (
    importlib.util.module_from_spec(spec)
)

spec.loader.exec_module(
    template_builder_module
)

TemplateBuilder = (
    template_builder_module.TemplateBuilder
)


# ============================================================
# TEMPLATE CREATION
# ============================================================

class TemplateCreation:

    def __init__(self):

        self.builder = TemplateBuilder()

        print("Template Creation ready.")


    # ========================================================
    # GET NUMBER OF REFERENCE IMAGES
    # ========================================================

    def get_image_count(self):

        while True:

            value = input(
                "\nHow many reference images? (4-5): "
            ).strip()

            try:

                count = int(value)

            except ValueError:

                print(
                    "Please enter a number."
                )

                continue

            if count < 4 or count > 5:

                print(
                    "Please use 4 or 5 reference images."
                )

                continue

            return count


    # ========================================================
    # CAPTURE REFERENCE IMAGES
    # ========================================================

    def capture_reference_images(self, count):

        reference_images = []

        print()
        print("================================")
        print("REFERENCE IMAGE CAPTURE")
        print("================================")

        print(
            f"Images required: {count}"
        )

        for index in range(1, count + 1):

            print()
            print(
                f"Reference image "
                f"{index}/{count}"
            )

            input(
                "Press ENTER to capture..."
            )

            image_path = capture_image()

            if image_path is None:

                raise RuntimeError(
                    f"Failed to capture "
                    f"reference image {index}."
                )

            reference_images.append(
                image_path
            )

            print(
                f"Captured: {image_path}"
            )

        return reference_images


    # ========================================================
    # CREATE TEMPLATE
    # ========================================================

    def create(self):

        print()
        print("================================")
        print("       CREATE NEW TEMPLATE")
        print("================================")

        template_name = input(
            "\nEnter template name: "
        ).strip()

        if not template_name:

            print(
                "Template name cannot be empty."
            )

            return None


        # ----------------------------------------------------
        # Number of reference images
        # ----------------------------------------------------

        count = self.get_image_count()


        # ----------------------------------------------------
        # Capture reference images
        # ----------------------------------------------------

        reference_images = (
            self.capture_reference_images(
                count
            )
        )


        # ----------------------------------------------------
        # Build template
        # ----------------------------------------------------

        print()
        print("================================")
        print("       BUILDING TEMPLATE")
        print("================================")

        template_data = (
            self.builder.create_template(
                template_name=template_name,
                reference_images=reference_images
            )
        )


        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        print()
        print("================================")
        print("       TEMPLATE COMPLETE")
        print("================================")

        print(
            f"Template: "
            f"{template_data['template_name']}"
        )

        print(
            f"Reference images: "
            f"{template_data['reference_image_count']}"
        )

        print(
            f"Expected components: "
            f"{template_data['component_count']}"
        )

        return template_data


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    try:

        creator = TemplateCreation()

        creator.create()

    except KeyboardInterrupt:

        print(
            "\nTemplate creation cancelled."
        )

    except Exception as error:

        print(
            f"\nTemplate creation failed: "
            f"{error}"
        )
