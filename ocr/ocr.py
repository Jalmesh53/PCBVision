import cv2
import pytesseract
import numpy as np
from pathlib import Path


class ComponentOCR:
    def __init__(self):
        self.config = "--psm 11"

    def _preprocess_variants(self, image):
        """
        Generate multiple lightweight preprocessing variants
        optimized for small PCB/component text.
        """

        # 1. Upscale
        image = cv2.resize(
            image,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        # 2. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 3. Contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(gray)

        # 4. Sharpen
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        sharpened = cv2.filter2D(
            enhanced,
            -1,
            kernel
        )

        # 5. Otsu threshold
        _, otsu = cv2.threshold(
            sharpened,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 6. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        return [
            enhanced,
            sharpened,
            otsu,
            adaptive
        ]

    def _clean_text(self, text):
        """Clean obvious OCR noise."""

        if not text:
            return ""

        text = text.replace("\n", " ")
        text = " ".join(text.split())

        return text.strip()

    def _score_text(self, text):
        """
        Simple OCR quality score.
        PCB text usually contains alphanumeric characters.
        """

        if not text:
            return 0

        alphanumeric = sum(
            c.isalnum()
            for c in text
        )

        length = len(text)

        if length == 0:
            return 0

        ratio = alphanumeric / length

        # Prefer useful alphanumeric text,
        # but penalize extremely long noisy outputs.
        score = alphanumeric * ratio

        if length > 150:
            score *= 0.5

        return score

    def extract_text(self, image_path):
        """
        Extract text from a cropped component image.

        Returns:
            str: best OCR result
        """

        image_path = Path(image_path)

        if not image_path.exists():
            print(f"OCR image not found: {image_path}")
            return ""

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Unable to read OCR image: {image_path}")
            return ""

        try:
            variants = self._preprocess_variants(image)

            results = []

            for variant in variants:

                # Standard OCR
                text = pytesseract.image_to_string(
                    variant,
                    config=self.config
                )

                text = self._clean_text(text)

                if text:
                    score = self._score_text(text)

                    results.append({
                        "text": text,
                        "score": score
                    })

                # Alternative sparse-text mode
                text_sparse = pytesseract.image_to_string(
                    variant,
                    config="--psm 6"
                )

                text_sparse = self._clean_text(text_sparse)

                if text_sparse:
                    score = self._score_text(text_sparse)

                    results.append({
                        "text": text_sparse,
                        "score": score
                    })

            if not results:
                return ""

            # Select highest-scoring result
            best = max(
                results,
                key=lambda x: x["score"]
            )

            return best["text"]

        except Exception as e:
            print(f"OCR failed: {e}")
            return ""


if __name__ == "__main__":
    print("Component OCR module ready.")
