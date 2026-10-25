
# PCBVision: An Embedded Explainable AI-Based Automated Optical Inspection System for PCB

**An end-to-end hardware and software platform for automated Printed Circuit Board (PCB) inspection using computer vision, explainable AI, and real-time hardware feedback.**

PCBVision combines a Raspberry Pi–based inspection engine with a Windows-hosted backend and web dashboard. It captures or receives PCB images, detects electronic components, evaluates available inspection checks, generates explainability outputs, stores inspection records, and presents the results through a centralized interface.

The system is designed to connect physical inspection hardware, machine-learning inference, inspection decision logic, database-backed records, and a user-facing dashboard into one workflow.

**Repository:** https://github.com/Jalmesh53/PCBVision

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [End-to-End Workflow](#end-to-end-workflow)
- [Inspection and Decision Logic](#inspection-and-decision-logic)
- [Explainable AI (XAI)](#explainable-ai-xai)
- [Hardware and Software Requirements](#hardware-and-software-requirements)
- [Repository Guide](#repository-guide)
- [Installation and Setup](#installation-and-setup)
- [Running the System](#running-the-system)
- [Inspection Modes](#inspection-modes)
- [Inspection Results and Data Flow](#inspection-results-and-data-flow)
- [API and Backend Integration](#api-and-backend-integration)
- [Testing and Troubleshooting](#testing-and-troubleshooting)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

PCB inspection is an important part of electronics manufacturing and quality control. Manual inspection can be time-consuming, difficult to standardize, and dependent on the operator's experience.

PCBVision aims to improve the inspection workflow by integrating computer vision, structured verification, AI-assisted explanations, and hardware-level status feedback.

The platform contains two primary computing environments:

### 1. Raspberry Pi inspection system

The Raspberry Pi performs the physical inspection workflow. It interfaces with the camera or receives an uploaded image, runs the detection and verification pipeline, generates explainability outputs, and controls inspection hardware such as status indicators.

### 2. Windows backend and dashboard

The Windows system provides the user-facing inspection interface and backend services. It communicates with the Raspberry Pi, processes the returned inspection data, stores inspection records in a database, and exposes results to the dashboard.

Together, these components provide a complete workflow from image acquisition to inspection result presentation.

## Key Features

- **AI-based component detection:** Uses a YOLO-based object detection pipeline to identify components in PCB images.
- **Multiple image input methods:** Supports camera-based inspection and image upload, with template selection available for reference-based verification.
- **X-MCCV verification:** Evaluates available component checks, including presence, count, position, identity, orientation, polarity, size/shape, and spacing.
- **Template-based inspection:** Allows detected components to be compared against expected reference information when a template is selected.
- **Detection-only inspection:** Supports general inspection without a selected reference template.
- **Explainable AI:** Produces Grad-CAM heatmaps and overlays to visualize image regions relevant to model predictions, with AI-generated explanations where available.
- **Inspection decision engine:** Consolidates applicable check results into an overall inspection status.
- **Hardware feedback:** Supports inspection-status signaling through connected hardware, including LED feedback.
- **Centralized dashboard:** Provides a graphical interface for starting inspections and viewing results.
- **Inspection history:** Stores inspection records for later review.
- **Database integration:** Persists inspection metadata, decisions, and associated information through the Windows backend.
- **Quality reporting:** Provides a foundation for reviewing inspection outcomes and generating reports.

---

## System Architecture

```text
                         PCBVision System
                                |
                +---------------+---------------+
                |                               |
        Raspberry Pi System              Windows System
                |                               |
        Camera / Image Input             Web Dashboard
                |                               |
        Image Acquisition                Backend API
                |                               |
        YOLO Detection                   Pi Communication
                |                               |
        OCR Processing                   Result Processing
                |                               |
        X-MCCV Verification              Database Operations
                |                               |
        Decision Engine                  Inspection History
                |                               |
        XAI Generation                   Dashboard Response
                |                               |
        Hardware Feedback <---------- Inspection Status
                |
        Inspection Result
```

### Architectural responsibilities

| Component | Responsibility |
|---|---|
| Camera / image input | Supplies the PCB image to inspect |
| YOLO detector | Detects trained object classes and returns detection metadata |
| OCR | Extracts text from image regions where applicable |
| X-MCCV | Performs component checks using available reference data and measurements |
| Decision engine | Consolidates check results into an overall status |
| XAI module | Generates visual explanations and optional AI-generated analysis |
| Raspberry Pi hardware interface | Provides physical inspection feedback |
| Windows API | Coordinates requests, processes responses, and serves inspection data |
| Database | Stores inspection records and related metadata |
| Dashboard | Displays inspection status, visual outputs, and history |

The Raspberry Pi and Windows backend have separate responsibilities. Successful inference on the Raspberry Pi does not, by itself, guarantee that the Windows API has successfully stored or returned the result to the dashboard.

---

## End-to-End Workflow

### Step 1 — Image acquisition

An inspection begins when the operator chooses an input method:

- Capture an image through the connected camera.
- Upload an existing PCB image.
- Select a reference template when template-based inspection is required.

The selected image becomes the input to the inspection pipeline.

### Step 2 — YOLO object detection

The detection engine processes the image and returns detections containing fields such as:

- Class ID and class name
- Confidence score
- Bounding-box coordinates
- Component center and size measurements
- Normalized position and size, where available

The detection output is also used to generate an annotated image for visual review.

Detection confidence describes the model's confidence in a predicted class. It does not independently prove that a component or the complete PCB is defect-free.

### Step 3 — OCR processing

The OCR stage extracts text from the image where applicable.

OCR output can support the interpretation of visible markings and inspection context. Its usefulness depends on image resolution, focus, lighting, component markings, and the OCR implementation.

### Step 4 — X-MCCV verification

The X-MCCV verification module evaluates available inspection checks.

Depending on the inspection mode and available reference information, checks can include:

- **Presence:** Whether an expected component is detected.
- **Count:** Whether the detected component count matches the expected count.
- **Position:** Whether the component is within the configured positional tolerance.
- **Identity:** Whether the detected class matches the expected component class.
- **Orientation:** Whether the measured orientation satisfies the expected tolerance.
- **Polarity:** Whether polarity matches the reference, where applicable.
- **Size/shape:** Whether measured dimensions satisfy the configured tolerance.
- **Spacing:** Whether spacing satisfies the applicable reference criteria.

Not every check is available for every component. Checks requiring expected values cannot provide a meaningful reference comparison when those values are unavailable.

### Step 5 — Inspection decision

The decision engine evaluates the available check results and derives component-level and overall inspection statuses.

The decision stage also integrates recognized PCB defect detections where configured in the main inspection pipeline.

The resulting status is included in the inspection response and is used by the rest of the system to present the outcome.

### Step 6 — Explainable AI

The XAI stage can generate:

- Grad-CAM heatmaps
- Image overlays
- Model-derived class information
- AI-generated explanation or recommendation fields, when available

The heatmap and overlay are saved as inspection artifacts. Optional language-model analysis may depend on network availability, configuration, and successful processing.

XAI outputs explain aspects of model behavior; they are not a substitute for validated electrical testing or complete PCB quality verification.

### Step 7 — Hardware feedback

The Raspberry Pi can use the inspection result to activate the configured hardware feedback.

For example, the current implementation has reported a green LED for a successful inspection. Exact signal behavior depends on the connected hardware and implementation.

### Step 8 — Windows backend processing

The Windows backend receives the Raspberry Pi response and extracts the relevant inspection data, including:

- Inspection status and decision reason
- Component detections
- X-MCCV results
- XAI results
- Image and artifact paths
- Inspection metadata

It then performs database operations and prepares a response for the dashboard.

### Step 9 — Dashboard presentation

The frontend displays the completed inspection, including the status, component results, visual artifacts, and relevant explanation fields.

A complete end-to-end inspection requires all necessary stages to succeed: inference, response transmission, database processing, and frontend rendering.

---

## Inspection and Decision Logic

PCBVision distinguishes between detection, verification, and the overall inspection decision.

### Component-level statuses

| Status | Meaning |
|---|---|
| `PASS` | The component's applicable checks passed under the configured decision rules |
| `FAIL` | At least one applicable check failed |
| `UNVERIFIED` | The available evidence was insufficient to produce an evaluable result |

### Overall inspection status

| Status | Interpretation |
|---|---|
| `PASS` | The configured decision rules accepted the available inspection evidence |
| `FAIL` | An applicable check failed or a recognized defect triggered the configured failure rule |
| `UNVERIFIED` | The available checks did not provide sufficient evidence for a definitive decision |

The precise meaning of `PASS` depends on the active inspection mode and the checks that were actually performed.

**Important:** A detection-only result must not be presented as proof that an entire PCB is defect-free. A board-level quality claim requires appropriate defect detection, verification criteria, and validation of the complete inspection process.

---

## Explainable AI (XAI)

PCBVision includes an XAI stage to improve visibility into model predictions.

### Grad-CAM visualization

Grad-CAM uses gradients associated with a target model output to create a coarse spatial explanation of image regions that influenced that prediction.

The system can save:

- `heatmap.jpg` — the generated activation heatmap.
- `xai_overlay.jpg` — the heatmap combined with the input image.

These files are associated with an inspection result and can be presented in the dashboard.

### AI-generated explanation

Where configured, the XAI stage may also produce structured fields such as:

- Defect description
- Location
- Explanation
- Recommendation
- Language-model status and model metadata

These fields may be absent when no explanation is available or when the optional AI service fails. Missing explanation data should not automatically invalidate an otherwise completed inspection unless the application explicitly requires it.

---

## Hardware and Software Requirements

### Hardware

| Component | Purpose |
|---|---|
| Raspberry Pi | Runs the inspection engine |
| Compatible camera | Captures PCB images |
| Status LEDs or configured indicators | Provides physical inspection feedback |
| Windows computer | Hosts the backend and dashboard |
| Network connection | Enables communication between the Pi and Windows system |
| PCB samples | Used for inspection and testing |

Additional hardware requirements depend on the camera interface, GPIO configuration, and any peripherals enabled in the project.

### Software

- Raspberry Pi OS and Python environment
- Windows and Python virtual environment
- YOLO-compatible model and inference dependencies
- OpenCV and image-processing dependencies
- OCR dependencies
- FastAPI and Uvicorn for the Windows API
- SQLAlchemy and the configured database driver
- Frontend dependencies required by the dashboard
- Optional external AI service configuration for language-model explanations

Use the dependency files and configuration already present in each environment as the authoritative installation reference.

---

## Repository Guide

The repository contains the hardware-side inspection system and the Windows-side application. The following is a functional guide to the main modules discussed in the project.

> Verify exact filenames against the current repository tree before relying on this section as a literal directory listing.

### Raspberry Pi inspection modules

| Module / path | Responsibility |
|---|---|
| `main.py` | Coordinates the end-to-end inspection workflow |
| `detection/detector.py` | Runs object detection and extracts detection metadata |
| `verification/x_mccv.py` | Implements component verification checks |
| `decision/decision.py` | Determines component-level and overall inspection statuses |
| `xai/xai.py` | Generates explainability artifacts and related outputs |
| `templates/` | Contains template-related logic and reference handling |
| `data/input/` | Stores inspection input images |
| `data/output/` | Stores annotated detection outputs |
| `results/` | Stores inspection-specific artifacts, including XAI outputs |
| Hardware interface modules | Control configured indicators and inspection hardware |

### Windows backend modules

| Module / path | Responsibility |
|---|---|
| `backend/app/main.py` | FastAPI application and inspection API logic |
| `backend/models.py` | SQLAlchemy database models |
| `backend/report_builder.py` | Builds quality reports from stored inspection data |
| Backend dependency/configuration files | Define Python dependencies and application settings |
| Frontend source files | Render the inspection interface and display API responses |

### Recommended code navigation

When tracing an inspection:

1. Start at the Raspberry Pi entry point.
2. Follow the detector output into the X-MCCV module.
3. Follow verification results into the decision engine.
4. Trace the returned inspection dictionary.
5. Follow the Windows endpoint that calls the Pi.
6. Follow the database insert/update path.
7. Follow the API response into the frontend's result-rendering logic.

This path helps isolate whether a problem originates in image processing, decision logic, network communication, database persistence, or dashboard rendering.

---

## Installation and Setup

Because the project has separate Raspberry Pi and Windows environments, configure each environment independently.

### 1. Clone the repository

On the Windows computer:

```bash
git clone https://github.com/Jalmesh53/PCBVision.git
cd PCBVision
```

### 2. Set up the Windows backend

Open PowerShell in the backend directory.

Create and activate a virtual environment if one does not already exist:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies using the dependency file provided by the backend:

```powershell
pip install -r requirements.txt
```

If the backend uses a different dependency filename, use that file instead.

Configure the database connection, Raspberry Pi address, and any required service credentials using the project's configuration files or environment variables. Do not commit credentials to source control.

### 3. Set up the Raspberry Pi

Connect to the Raspberry Pi and enter the project directory.

Create and activate a Python virtual environment using the method supported by the installed Raspberry Pi OS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies from the Raspberry Pi's requirements file:

```bash
pip install -r requirements.txt
```

The exact dependency filename and installation procedure should match the files maintained in the Raspberry Pi project.

Verify that:

- The camera is accessible.
- The YOLO model weights are available.
- The configured input and output directories are writable.
- The hardware interface is configured correctly.
- The Raspberry Pi API address and port are reachable from Windows.

Some camera and GPIO packages may need to be installed using the operating system's package manager instead of pip.

---

## Running the System

### Start the Raspberry Pi inspection service

Activate the Pi's virtual environment and start the inspection service using the project's configured startup command.

For a project whose entry point is `main.py`, the command may be:

```bash
python main.py
```

Use this command only if `main.py` is the configured server or application entry point in the current deployment.

Confirm that the service is running and that the camera is available before proceeding.

### Start the Windows backend

Open PowerShell in the Windows backend directory, activate its virtual environment, and run:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

This command matches the backend startup configuration used during development.

The API will be available locally at:

```text
http://localhost:8000
```

If the frontend runs on another device, configure its API base URL appropriately. The hostname `localhost` refers to the machine running the browser, not automatically to the Windows server from every client device.

### Open the dashboard

Start the frontend using the command defined by its package configuration or startup instructions.

Open the configured dashboard URL in a browser, verify that the API is reachable, and confirm that the camera connection indicator reports the expected state.

### Run an inspection

1. Select **Camera** or **Upload**.
2. Select a template only when reference-based verification is intended.
3. Start the inspection.
4. Wait for detection, verification, decision, and XAI processing to finish.
5. Review the returned status and explanation.
6. Confirm that the completed inspection appears in history and the database.

---

## Inspection Modes

### Detection-only inspection

Used when no reference template is selected.

The system runs detection and evaluates the checks available in that mode. Its ability to determine board quality is limited by the available evidence and the configured decision rules.

### Template-based inspection

Used when a reference template is selected.

The system can compare detected components against expected component data, including applicable counts, positions, identities, and measurements.

Template verification depends on the quality and completeness of the reference data and the correctness of the comparison criteria.

### Camera inspection

Captures an image through the configured camera and submits it to the inspection pipeline.

### Uploaded-image inspection

Processes a user-provided image. Image quality and resolution can affect detection, OCR, measurement, and XAI outputs.

---

## Inspection Results and Data Flow

An inspection response may contain the following types of information.

| Field | Description |
|---|---|
| `inspection_id` | Identifier for the inspection |
| `status` | Overall inspection status |
| `decision` | Structured decision information and reason |
| `detections` | Detected objects and associated metadata |
| `xmccv` | Component verification results |
| `detected_defects` | Recognized PCB defect detections, where configured |
| `xai` | Explainability results and generated artifact paths |
| `image_path` | Input image location |
| `detection_image` | Annotated detection image location |

The exact response schema depends on the active backend implementation and API version.

The Windows backend persists inspection information and returns data for dashboard rendering. The frontend should handle missing optional fields without discarding a valid inspection decision.

A robust integration should distinguish:

- Inference failure
- Verification failure
- A valid `FAIL` inspection
- An `UNVERIFIED` inspection
- Database failure
- API communication failure
- Optional XAI failure
- Frontend rendering failure

These conditions are not equivalent and should not be represented by the same user-facing message.

---

## API and Backend Integration

The Windows backend exposes API endpoints for the dashboard and communicates with the Raspberry Pi inspection service.

The inspection workflow used during development includes:

```text
POST /api/inspection/run
```

This endpoint coordinates the inspection request and processing of the Raspberry Pi response.

The exact request body, authentication requirements, and response schema should be verified against the current FastAPI implementation and its API documentation.

When debugging integration issues, verify the following sequence:

1. The dashboard sends the inspection request to the correct Windows API.
2. The Windows backend successfully communicates with the Raspberry Pi.
3. The Pi returns a valid inspection payload.
4. The Windows backend extracts the status, detections, verification results, and XAI fields.
5. The database operation completes.
6. The API returns a valid response.
7. The frontend reads the returned fields and renders the result.

If the Pi reports `PASS` but the dashboard remains at an awaiting state, inspect the Windows backend response and traceback before changing the Pi's decision engine.

---

## Testing and Troubleshooting

### 1. The dashboard remains at "Awaiting Inspection"

Check the browser's Network panel and inspect the response from the inspection API.

Confirm that the request succeeded and that the response includes the expected inspection status.

### 2. The Pi returns a result but Windows reports an API error

Inspect the Windows backend terminal for the full Python traceback. Check the endpoint's response parsing, exception handling, database transaction, and response construction.

### 3. An inspection returns `UNVERIFIED`

Review the applicable component check results, verification mode, and decision reason.

`UNVERIFIED` can be a legitimate outcome when the system lacks evaluable checks. It should not be changed to `PASS` merely to make the dashboard appear successful.

### 4. XAI output is missing

Check whether the XAI stage completed and whether the expected artifact paths and explanation fields were returned.

Optional XAI failure should be handled separately from the inspection decision unless the configured workflow explicitly requires XAI completion.

### 5. Images do not appear in the dashboard

Verify that the image paths returned by the Pi are accessible to the Windows backend or are converted into valid image URLs. A path on the Raspberry Pi's filesystem is not automatically a valid path on Windows.

### 6. The database operation fails

Inspect the database exception, transaction rollback, connection configuration, and SQLAlchemy model/schema compatibility.

### 7. The hardware indicator does not match the displayed status

Verify the status value received by the hardware control code, the configured GPIO pin mapping, and the indicator's active-high or active-low behavior.

### Run the decision-engine tests

For the Raspberry Pi project, activate its environment and run:

```bash
python decision/decision.py
```

This runs the decision-engine test cases if the file's built-in test section is present.

For broader tests, use the test commands and test files maintained by the repository.

---

## Limitations and Future Improvements

PCBVision is an integrated inspection platform, but its results depend on model training, camera quality, reference data, and validation of the inspection rules.

Potential improvements include:

- Expanding and balancing the training dataset.
- Improving defect-class coverage and detection reliability.
- Calibrating confidence thresholds against validation data.
- Validating measurements across different camera angles and image resolutions.
- Improving OCR robustness for small component markings.
- Adding stronger template validation and reference-data management.
- Separating detection confidence from verified board-quality status in the UI.
- Adding structured API schemas and consistent error responses.
- Improving frontend recovery from failed API or database requests.
- Adding automated integration tests across the Pi, Windows backend, database, and dashboard.
- Recording model versions and inspection configuration for reproducibility.
- Adding hardware diagnostics and more detailed inspection audit logs.

Future work should prioritize measurable inspection accuracy and reliable system integration rather than treating a successful model prediction as a guarantee of board quality.

---

## Contributing

Contributions are welcome.

A useful contribution workflow is:

1. Create a feature branch.
2. Make a focused change.
3. Run the relevant unit and integration tests.
4. Verify that existing inspection modes still behave correctly.
5. Document configuration or API changes.
6. Submit a pull request with a clear description of the change and its validation.

When modifying the decision engine, preserve the distinction between `PASS`, `FAIL`, and `UNVERIFIED`. When modifying the API, preserve the distinction between inspection outcomes and infrastructure errors.

---

## License

Check the repository's `LICENSE` file for the applicable license and redistribution terms.

If no license file is present, clarify the project's licensing terms before presenting the source code as open-source software.

---

## Acknowledgements

PCBVision brings together embedded computing, object detection, image processing, component verification, explainable AI, backend development, database persistence, and hardware integration.

The project demonstrates an end-to-end engineering approach to PCB inspection, from physical image acquisition to a software interface for reviewing inspection outcomes.
