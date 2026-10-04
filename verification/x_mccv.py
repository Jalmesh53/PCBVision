import math


class XMCCV:

    def __init__(
        self,
        position_tolerance=0.10,
        orientation_tolerance=20,
        size_tolerance=0.20,
        spacing_tolerance=0.20
    ):
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.size_tolerance = size_tolerance
        self.spacing_tolerance = spacing_tolerance

    # ============================================================
    # PRESENCE
    # ============================================================

    def verify_presence(self, detection, expected):

        if detection is None:
            return False

        return (
            detection.get("class_name")
            == expected.get("class_name")
        )

    # ============================================================
    # COUNT
    # ============================================================

    def verify_count(self, detections, expected):

        expected_count = expected.get("count")

        if expected_count is None:
            return True

        class_name = expected.get("class_name")

        actual_count = sum(
            1
            for detection in detections
            if detection.get("class_name") == class_name
        )

        return actual_count == expected_count

    # ============================================================
    # POSITION
    # ============================================================

    def verify_position(self, detection, expected):
        """
        Compare normalized component positions.

        Normalized coordinates are used instead of pixel
        coordinates so that reference and inspection images
        can have different resolutions.
        """

        expected_center = expected.get("normalized_center")

        if expected_center is None:
            return True

        actual_center = detection.get("normalized_center")

        if actual_center is None:
            return False

        if (
            "x" not in expected_center
            or "y" not in expected_center
        ):
            return True

        dx = (
            actual_center["x"]
            - expected_center["x"]
        )

        dy = (
            actual_center["y"]
            - expected_center["y"]
        )

        distance = math.sqrt(
            dx * dx + dy * dy
        )

        return distance <= self.position_tolerance

    # ============================================================
    # IDENTITY
    # ============================================================

    def verify_identity(self, detection, expected):

        expected_class = expected.get("class_name")
        actual_class = detection.get("class_name")

        return actual_class == expected_class

    # ============================================================
    # ORIENTATION
    # ============================================================

    def verify_orientation(self, detection, expected):

        expected_orientation = expected.get(
            "orientation"
        )

        if expected_orientation is None:
            return True

        actual_orientation = detection.get(
            "orientation"
        )

        if actual_orientation is None:
            return False

        difference = abs(
            actual_orientation
            - expected_orientation
        )

        difference = min(
            difference,
            360 - difference
        )

        return (
            difference
            <= self.orientation_tolerance
        )

    # ============================================================
    # POLARITY
    # ============================================================

    def verify_polarity(self, detection, expected):

        expected_polarity = expected.get(
            "polarity"
        )

        if expected_polarity is None:
            return True

        actual_polarity = detection.get(
            "polarity"
        )

        if actual_polarity is None:
            return False

        return (
            actual_polarity
            == expected_polarity
        )

    # ============================================================
    # SIZE / SHAPE
    # ============================================================

    def verify_size_shape(self, detection, expected):

        expected_size = expected.get(
            "size_shape"
        )

        if expected_size is None:
            return True

        actual_size = detection.get(
            "normalized_size"
        )

        if actual_size is None:
            return False

        expected_width = expected_size.get(
            "width"
        )

        expected_height = expected_size.get(
            "height"
        )

        actual_width = actual_size.get(
            "width"
        )

        actual_height = actual_size.get(
            "height"
        )

        if (
            expected_width is None
            or expected_height is None
            or actual_width is None
            or actual_height is None
        ):
            return True

        width_difference = abs(
            actual_width - expected_width
        )

        height_difference = abs(
            actual_height - expected_height
        )

        width_tolerance = (
            expected_width
            * self.size_tolerance
        )

        height_tolerance = (
            expected_height
            * self.size_tolerance
        )

        return (
            width_difference
            <= width_tolerance
            and
            height_difference
            <= height_tolerance
        )

    # ============================================================
    # SPACING
    # ============================================================

    def verify_spacing(self, detection, expected):
        """
        Spacing verification will become active when
        reference spacing data is available.
        """

        expected_spacing = expected.get(
            "spacing"
        )

        if expected_spacing is None:
            return True

        actual_spacing = detection.get(
            "spacing"
        )

        if actual_spacing is None:
            return False

        return True

    # ============================================================
    # FIND BEST MATCH
    # ============================================================

    def find_best_detection(
        self,
        detections,
        expected
    ):

        expected_class = expected.get(
            "class_name"
        )

        expected_center = expected.get(
            "normalized_center"
        )

        best_index = None
        best_distance = float("inf")

        for index, detection in enumerate(
            detections
        ):

            if (
                detection.get("class_name")
                != expected_class
            ):
                continue

            actual_center = detection.get(
                "normalized_center"
            )

            if (
                actual_center is None
                or expected_center is None
            ):
                continue

            dx = (
                actual_center["x"]
                - expected_center["x"]
            )

            dy = (
                actual_center["y"]
                - expected_center["y"]
            )

            distance = math.sqrt(
                dx * dx + dy * dy
            )

            if distance < best_distance:
                best_distance = distance
                best_index = index

        return (
            best_index,
            best_distance
        )

    # ============================================================
    # MAIN VERIFICATION
    # ============================================================

    def verify(
        self,
        template,
        detections,
        expected_components
    ):

        results = []

        used_detection_indices = set()

        # ========================================================
        # EXPECTED COMPONENTS
        # ========================================================

        for expected in expected_components:

            expected_class = expected.get(
                "class_name"
            )

            matching_index, match_distance = (
                self.find_best_detection(
                    detections,
                    expected
                )
            )

            # ----------------------------------------------------
            # COMPONENT MISSING
            # ----------------------------------------------------

            if matching_index is None:

                results.append({

                    "reference_id":
                        expected.get(
                            "reference_id"
                        ),

                    "class_name":
                        expected_class,

                    # NEW:
                    # Expected location from the template.
                    "expected_normalized_center":
                        expected.get(
                            "normalized_center"
                        ),

                    "presence":
                        False,

                    "count":
                        self.verify_count(
                            detections,
                            expected
                        ),

                    "position":
                        False,

                    "identity":
                        False,

                    "orientation":
                        True,

                    "polarity":
                        True,

                    "size_shape":
                        True,

                    "spacing":
                        True,

                    "component_pass":
                        False,

                    "unexpected":
                        False,

                    "match_distance":
                        None,

                    "reason":
                        "Component missing"
                })

                continue

            # ----------------------------------------------------
            # COMPONENT FOUND
            # ----------------------------------------------------

            used_detection_indices.add(
                matching_index
            )

            detection = detections[
                matching_index
            ]

            presence = self.verify_presence(
                detection,
                expected
            )

            count = self.verify_count(
                detections,
                expected
            )

            position = self.verify_position(
                detection,
                expected
            )

            identity = self.verify_identity(
                detection,
                expected
            )

            orientation = (
                self.verify_orientation(
                    detection,
                    expected
                )
            )

            polarity = (
                self.verify_polarity(
                    detection,
                    expected
                )
            )

            size_shape = (
                self.verify_size_shape(
                    detection,
                    expected
                )
            )

            spacing = (
                self.verify_spacing(
                    detection,
                    expected
                )
            )

            component_pass = all([
                presence,
                count,
                position,
                identity,
                orientation,
                polarity,
                size_shape,
                spacing
            ])

            if component_pass:
                reason = (
                    "All applicable criteria passed"
                )

            else:

                failed = []

                if not presence:
                    failed.append("Presence")

                if not count:
                    failed.append("Count")

                if not position:
                    failed.append("Position")

                if not identity:
                    failed.append("Identity")

                if not orientation:
                    failed.append("Orientation")

                if not polarity:
                    failed.append("Polarity")

                if not size_shape:
                    failed.append("Size/Shape")

                if not spacing:
                    failed.append("Spacing")

                reason = (
                    "Failed: "
                    + ", ".join(failed)
                )

            results.append({

                "reference_id":
                    expected.get(
                        "reference_id"
                    ),

                "class_name":
                    expected_class,

                "presence":
                    presence,

                "count":
                    count,

                "position":
                    position,

                "identity":
                    identity,

                "orientation":
                    orientation,

                "polarity":
                    polarity,

                "size_shape":
                    size_shape,

                "spacing":
                    spacing,

                "component_pass":
                    component_pass,

                "unexpected":
                    False,

                "match_distance":
                    match_distance,

                "reason":
                    reason
            })

        # ========================================================
        # UNEXPECTED COMPONENTS
        # ========================================================

        for index, detection in enumerate(
            detections
        ):

            if index in used_detection_indices:
                continue

            results.append({

                "reference_id":
                    None,

                "class_name":
                    detection.get(
                        "class_name"
                    ),

                "presence":
                    False,

                "count":
                    False,

                "position":
                    False,

                "identity":
                    False,

                "orientation":
                    True,

                "polarity":
                    True,

                "size_shape":
                    True,

                "spacing":
                    True,

                "component_pass":
                    False,

                "unexpected":
                    True,

                "match_distance":
                    None,

                "reason":
                    "Unexpected component detected"
            })

        return results


# ================================================================
# MODULE TEST
# ================================================================

if __name__ == "__main__":

    print("X-MCCV module ready.")
    print("Template-based verification is available.")
