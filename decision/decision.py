
class DecisionEngine:

    CHECK_FIELDS = (
        "presence",
        "count",
        "position",
        "identity",
        "orientation",
        "polarity",
        "size_shape",
        "spacing",
    )

    def _get_check_status(self, value):
        """Return PASS, FAIL, or None if the check is unavailable."""

        if value is None:
            return None

        if isinstance(value, dict):
            return self._get_check_status(value.get("status"))

        if isinstance(value, bool):
            return "PASS" if value else "FAIL"

        if isinstance(value, str):
            normalized = value.strip().upper()

            if normalized in ("PASS", "TRUE"):
                return "PASS"

            if normalized in ("FAIL", "FALSE"):
                return "FAIL"

            if normalized in (
                "N/A",
                "NA",
                "NONE",
                "UNVERIFIED",
                "",
            ):
                return None

        # Raw measurements and class names are not check results.
        return None

    def _evaluate_component(self, result):
        """Evaluate a component using available check results."""

        statuses = []

        for field in self.CHECK_FIELDS:
            status = self._get_check_status(result.get(field))

            if status is not None:
                statuses.append(status)

        # In template-free mode, do not rely on component_pass
        # from an earlier or externally derived decision.
        if result.get("verification_mode") != "detection_only":
            existing = result.get("component_pass")

            if isinstance(existing, bool):
                statuses.append(
                    "PASS" if existing else "FAIL"
                )

        if "FAIL" in statuses:
            return "FAIL"

        if "PASS" in statuses:
            return "PASS"

        return "UNVERIFIED"

    def evaluate(self, x_mccv_results):
        """Evaluate component results and determine inspection status."""

        if not x_mccv_results:
            return {
                "overall_status": "UNVERIFIED",
                "reason": "No component detections or checks were available",
                "passed_components": 0,
                "failed_components": 0,
                "unverified_components": 0,
            }

        passed_components = 0
        failed_components = 0
        unverified_components = 0

        for result in x_mccv_results:
            component_status = self._evaluate_component(result)

            result["component_status"] = component_status

            if component_status == "PASS":
                result["component_pass"] = True
                passed_components += 1

            elif component_status == "FAIL":
                result["component_pass"] = False
                failed_components += 1

            else:
                result["component_pass"] = None
                unverified_components += 1

        if failed_components > 0:
            overall_status = "FAIL"
            reason = (
                "One or more components failed an applicable check"
            )

        elif unverified_components > 0:
            overall_status = "UNVERIFIED"
            reason = (
                "One or more components had no evaluable PASS/FAIL check"
            )

        elif passed_components > 0:
            overall_status = "PASS"
            reason = (
                "All detected components passed the available checks"
            )

        else:
            overall_status = "UNVERIFIED"
            reason = (
                "No component had an evaluable PASS/FAIL check"
            )

        return {
            "overall_status": overall_status,
            "reason": reason,
            "passed_components": passed_components,
            "failed_components": failed_components,
            "unverified_components": unverified_components,
        }


if __name__ == "__main__":

    engine = DecisionEngine()

    test_cases = [
        {
            "name": "Detection-only component found",
            "results": [
                {
                    "class_name": "resistor",
                    "presence": True,
                    "identity": "resistor",
                    "verification_mode": "detection_only",
                    "component_pass": None,
                }
            ],
            "expected": "PASS",
        },
        {
            "name": "Explicit failed position check",
            "results": [
                {
                    "class_name": "ESP32",
                    "presence": True,
                    "identity": "ESP32",
                    "position": False,
                    "component_pass": None,
                }
            ],
            "expected": "FAIL",
        },
        {
            "name": "No evaluable checks",
            "results": [
                {
                    "class_name": "ESP32",
                    "presence": None,
                    "identity": None,
                    "orientation": None,
                    "polarity": None,
                    "component_pass": None,
                }
            ],
            "expected": "UNVERIFIED",
        },
        {
            "name": "All applicable checks pass",
            "results": [
                {
                    "class_name": "ESP32",
                    "presence": "PASS",
                    "position": "PASS",
                    "identity": "PASS",
                    "component_pass": None,
                }
            ],
            "expected": "PASS",
        },
    ]

    all_passed = True

    for test in test_cases:
        decision = engine.evaluate(test["results"])

        actual = decision["overall_status"]
        expected = test["expected"]
        passed = actual == expected

        if not passed:
            all_passed = False

        print(f"\nTest: {test['name']}")
        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")
        print(f"Result:   {'PASS' if passed else 'FAIL'}")

    print(
        "\nAll tests passed."
        if all_passed
        else "\nSome tests failed."
    )
