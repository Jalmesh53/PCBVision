import json
import shutil
from datetime import datetime
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path("/home/pi4/PCBVISION")
TEMPLATES_DIR = BASE_DIR / "templates"

ACTIVE_TEMPLATE_FILE = (
    TEMPLATES_DIR /
    "active_template.json"
)


# ============================================================
# TEMPLATE MANAGER
# ============================================================

class TemplateManager:

    def __init__(self):

        TEMPLATES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


    # ========================================================
    # NAME SANITIZATION
    # ========================================================

    def _sanitize_name(
        self,
        template_name
    ):

        safe_name = "".join(
            c
            if c.isalnum()
            or c in "_- "
            else "_"
            for c in template_name
        ).strip()

        if not safe_name:

            raise ValueError(
                "Invalid template name."
            )

        return safe_name


    # ========================================================
    # TEMPLATE PATHS
    # ========================================================

    def _template_dir(
        self,
        template_name
    ):

        return (
            TEMPLATES_DIR /
            template_name
        )


    def _template_json(
        self,
        template_name
    ):

        return (
            self._template_dir(
                template_name
            )
            /
            "template.json"
        )


    # ========================================================
    # CREATE TEMPLATE
    # ========================================================

    def create_template(
        self,
        template_name,
        reference_images,
        expected_components
    ):

        if not template_name:

            raise ValueError(
                "Template name cannot be empty."
            )

        if not reference_images:

            raise ValueError(
                "At least one reference image "
                "is required."
            )

        safe_name = self._sanitize_name(
            template_name
        )

        template_dir = (
            self._template_dir(
                safe_name
            )
        )

        images_dir = (
            template_dir /
            "images"
        )

        if template_dir.exists():

            raise FileExistsError(
                f"Template already exists: "
                f"{safe_name}"
            )

        images_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        stored_images = []

        for index, image_path in enumerate(
            reference_images,
            start=1
        ):

            image_path = Path(
                image_path
            )

            if not image_path.exists():

                raise FileNotFoundError(
                    f"Reference image not found: "
                    f"{image_path}"
                )

            extension = (
                image_path.suffix.lower()
            )

            if not extension:

                extension = ".jpg"

            filename = (
                f"reference_{index:02d}"
                f"{extension}"
            )

            destination = (
                images_dir /
                filename
            )

            shutil.copy2(
                image_path,
                destination
            )

            stored_images.append(
                str(
                    Path("images") /
                    filename
                )
            )

        template_data = {

            "template_name":
                safe_name,

            "created_at":
                datetime.now().isoformat(),

            "reference_images":
                stored_images,

            "reference_image_count":
                len(stored_images),

            "component_count":
                len(expected_components),

            "components":
                expected_components
        }

        template_json = (
            self._template_json(
                safe_name
            )
        )

        with open(
            template_json,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                template_data,
                file,
                indent=4
            )

        print(
            f"Template created: "
            f"{safe_name}"
        )

        print(
            f"Reference images stored: "
            f"{len(stored_images)}"
        )

        print(
            f"Expected components: "
            f"{len(expected_components)}"
        )

        return template_data


    # ========================================================
    # LOAD TEMPLATE
    # ========================================================

    def load_template(
        self,
        template_name
    ):

        template_json = (
            self._template_json(
                template_name
            )
        )

        if not template_json.exists():

            raise FileNotFoundError(
                f"Template not found: "
                f"{template_name}"
            )

        with open(
            template_json,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)


    # ========================================================
    # LIST TEMPLATES
    # ========================================================

    def list_templates(self):

        if not TEMPLATES_DIR.exists():

            return []

        templates = []

        for directory in sorted(
            TEMPLATES_DIR.iterdir()
        ):

            if not directory.is_dir():

                continue

            template_json = (
                directory /
                "template.json"
            )

            if template_json.exists():

                templates.append(
                    directory.name
                )

        return templates


    # ========================================================
    # SET ACTIVE TEMPLATE
    # ========================================================

    def set_active_template(
        self,
        template_name
    ):

        safe_name = self._sanitize_name(
            template_name
        )

        template_json = (
            self._template_json(
                safe_name
            )
        )

        if not template_json.exists():

            raise FileNotFoundError(
                f"Template not found: "
                f"{safe_name}"
            )

        active_data = {

            "active_template":
                safe_name,

            "updated_at":
                datetime.now().isoformat()
        }

        with open(
            ACTIVE_TEMPLATE_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                active_data,
                file,
                indent=4
            )

        print(
            f"Active template set: "
            f"{safe_name}"
        )

        return safe_name


    # ========================================================
    # GET ACTIVE TEMPLATE
    # ========================================================

    def get_active_template(self):

        if not ACTIVE_TEMPLATE_FILE.exists():

            return None

        try:

            with open(
                ACTIVE_TEMPLATE_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            template_name = (
                data.get(
                    "active_template"
                )
            )

            if not template_name:

                return None

            # Make sure the template still exists.

            template_json = (
                self._template_json(
                    template_name
                )
            )

            if not template_json.exists():

                return None

            return template_name

        except Exception:

            return None


    # ========================================================
    # DELETE TEMPLATE
    # ========================================================

    def delete_template(
        self,
        template_name
    ):

        template_dir = (
            self._template_dir(
                template_name
            )
        )

        if not template_dir.exists():

            raise FileNotFoundError(
                f"Template not found: "
                f"{template_name}"
            )

        shutil.rmtree(
            template_dir
        )

        # If deleted template was active,
        # clear active template.

        active_template = (
            self.get_active_template()
        )

        if active_template == template_name:

            if ACTIVE_TEMPLATE_FILE.exists():

                ACTIVE_TEMPLATE_FILE.unlink()

        print(
            f"Template deleted: "
            f"{template_name}"
        )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    manager = TemplateManager()

    print(
        "Template Manager ready."
    )

    templates = (
        manager.list_templates()
    )

    print(
        "\nAvailable templates:"
    )

    if templates:

        for template in templates:

            print(
                f"  - {template}"
            )

    else:

        print(
            "  No templates created yet."
        )

    active_template = (
        manager.get_active_template()
    )

    print(
        "\nActive template:"
    )

    if active_template:

        print(
            f"  {active_template}"
        )

    else:

        print(
            "  None"
        )
