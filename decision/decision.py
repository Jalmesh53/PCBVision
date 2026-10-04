from pathlib import Path

BASE_DIR = Path("/home/pi4/PCBVISION")


class DecisionEngine:

    def evaluate(self, x_mccv_results):

        if not x_mccv_results:
            return {
                "overall_status": "FAIL",
                "reason": "No components verified",
                "passed_components": 0,
                "failed_components": 0
            }

        passed_components = 0
        failed_components = 0

        for result in x_mccv_results:

            if result.get("component_pass", False):
                passed_components += 1
            else:
                failed_components += 1

        if failed_components == 0:
            overall_status = "PASS"
            reason = "All verified components passed X-MCCV"
        else:
            overall_status = "FAIL"
            reason = "One or more components failed X-MCCV"

        return {
            "overall_status": overall_status,
            "reason": reason,
            "passed_components": passed_components,
            "failed_components": failed_components
        }


if __name__ == "__main__":

    test_result = [
        {
            "component_id": "U1",
            "class_name": "ESP32",
            "presence": {
                "status": "PASS"
            },
            "count": {
                "status": "PASS"
            },
            "position": {
                "status": "PASS"
            },
            "component_pass": True
        }
    ]

    engine = DecisionEngine()

    decision = engine.evaluate(test_result)

    print("\nDECISION ENGINE RESULT:")
    print(decision)
