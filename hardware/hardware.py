import time
from gpiozero import LED, Buzzer
from RPLCD.i2c import CharLCD


class HardwareController:

    def __init__(self):

        # -------------------------
        # LEDs
        # -------------------------
        self.green_led = LED(17)
        self.red_led = LED(27)

        # -------------------------
        # Buzzer
        # -------------------------
        self.buzzer = Buzzer(22)

        # -------------------------
        # LCD
        # -------------------------
        self.lcd = CharLCD(
            i2c_expander="PCF8574",
            address=0x27,
            port=1,
            cols=16,
            rows=2,
            charmap="A00"
        )

        self.all_off()
        self.ready()

    # -------------------------
    # LCD HELPER
    # -------------------------
    def lcd_display(self, line1="", line2=""):

        self.lcd.clear()

        # Center first line on 16x2 LCD
        line1 = line1.center(16)

        self.lcd.cursor_pos = (0, 0)
        self.lcd.write_string(line1[:16])

        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string(line2[:16])

    # -------------------------
    # LEDs / BUZZER
    # -------------------------
    def all_off(self):

        self.green_led.off()
        self.red_led.off()
        self.buzzer.off()

    # -------------------------
    # READY
    # -------------------------
    def ready(self):

        self.all_off()

        self.lcd_display(
            "PCBVISION",
            "Ready for Test"
        )

    # -------------------------
    # CAMERA
    # -------------------------
    def capturing_image(self):

        self.all_off()

        self.lcd_display(
            "PCBVISION",
            "Capturing Image"
        )

    def image_captured(self):

        self.all_off()

        self.lcd_display(
            "PCBVISION",
            "Image Captured"
        )

    # -------------------------
    # TEMPLATE CAPTURE
    # -------------------------
    def template_capture_progress(
        self,
        template_name,
        current_image,
        total_images,
    ):

        self.all_off()

        self.lcd_display(
            "TEMPLATE CAPTURE",
            f"Image {current_image}/{total_images}"
        )

    def template_building(self):

        self.all_off()

        self.lcd_display(
            "BUILDING TEMPLATE",
            "Please Wait..."
        )

    def template_saved(self, template_name):

        # Return to the default screen after template creation.
        self.ready()

    # -------------------------
    # UPLOAD
    # -------------------------
    def loading_image(self):

        self.all_off()

        self.lcd_display(
            "PCBVISION",
            "Loading Image"
        )

    def image_loaded(self):

        self.all_off()

        self.lcd_display(
            "PCBVISION",
            "Image Loaded"
        )

    # -------------------------
    # INSPECTION STAGES
    # -------------------------
    def validating_model(self):

        self.all_off()

        self.lcd_display(
            "PCBVISION",
            "Validating Model"
        )

    def reading_parts(self):

        self.all_off()

        self.lcd_display(
            "PCBVISION",
            "Reading Parts"
        )

    def applying_xmccv(self):

        self.all_off()

        self.lcd_display(
            "PCBVISION",
            "Applying X-MCCV"
        )

    def making_decision(self):

        self.all_off()

        self.lcd_display(
            "PCBVISION",
            "Making Decision"
        )

    def generating_xai(self):

        self.all_off()

        self.lcd_display(
            "PCBVISION",
            "Generating XAI"
        )

    # -------------------------
    # INSPECTION COMPLETE
    # -------------------------
    def inspection_complete(self):

        self.all_off()

        self.lcd_display(
            "PCBVISION",
            "Inspection Done"
        )

    # -------------------------
    # PASS
    # -------------------------
    def pass_signal(self):

        self.all_off()

        self.green_led.on()

        self.lcd_display(
            "PCBVISION",
            "RESULT: PASS"
        )

    # -------------------------
    # FAIL
    # -------------------------
    def fail_signal(self):

        self.all_off()

        self.red_led.on()

        # Keep the existing buzzer behavior.
        try:
            self.buzzer.on()
        except Exception:
            pass

        self.lcd_display(
            "PCBVISION",
            "RESULT: FAIL"
        )

    # -------------------------
    # ERROR
    # -------------------------
    def error(self, message="ERROR"):

        self.all_off()

        self.red_led.on()

        try:
            self.buzzer.on()
        except Exception:
            pass

        self.lcd_display(
            "PCBVISION",
            message
        )

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):

        self.all_off()
        self.ready()


# =========================================================
# HARDWARE TEST
# =========================================================

if __name__ == "__main__":

    hardware = HardwareController()

    print("LCD READY")
    time.sleep(3)

    print("CAPTURING IMAGE")
    hardware.capturing_image()
    time.sleep(2)

    print("IMAGE CAPTURED")
    hardware.image_captured()
    time.sleep(2)

    print("TEMPLATE CAPTURE PROGRESS")
    hardware.template_capture_progress(
        "MAIN_PCB",
        1,
        4
    )
    time.sleep(2)

    print("BUILDING TEMPLATE")
    hardware.template_building()
    time.sleep(2)

    print("TEMPLATE SAVED")
    hardware.template_saved("MAIN_PCB")
    time.sleep(2)

    print("VALIDATING MODEL")
    hardware.validating_model()
    time.sleep(2)

    print("READING PARTS")
    hardware.reading_parts()
    time.sleep(2)

    print("APPLYING X-MCCV")
    hardware.applying_xmccv()
    time.sleep(2)

    print("MAKING DECISION")
    hardware.making_decision()
    time.sleep(2)

    print("GENERATING XAI")
    hardware.generating_xai()
    time.sleep(2)

    print("INSPECTION COMPLETE")
    hardware.inspection_complete()
    time.sleep(2)

    print("PASS")
    hardware.pass_signal()
    time.sleep(3)

    print("FAIL")
    hardware.fail_signal()
    time.sleep(3)

    print("RESET")
    hardware.reset()

    print("Hardware integration test complete.")
