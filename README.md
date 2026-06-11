# JewelGuard AI - Jewelry Store Surveillance System

JewelGuard AI is a computer vision surveillance prototype designed for jewelry store security. The system uses OpenCV, YOLO person detection, face-covering detection, polygon-based security zones, motion analysis, wrist/case proximity detection, and a live React dashboard to identify suspicious activity near store entrances and jewelry display cases.

The goal of this project is **not** to classify every movement as criminal. Instead, the system separates normal customer movement from higher-risk behavior by combining multiple signals such as zone location, fast entrance movement, sudden hand/body motion near display cases, dwell time, face covering, and protected-case boundary interaction.

---

## Project Description

Jewelry stores are high-risk retail environments where security systems need to detect suspicious behavior quickly while avoiding false alarms from normal customers. Basic motion detection is not enough because normal walking, reflections, lighting changes, employee movement, and camera noise can all create motion.

JewelGuard AI improves on simple OpenCV motion detection by using context-aware computer vision. Instead of saying "motion equals suspicious," the system asks:

- Is a person inside a sensitive zone?
- Is the person moving unusually fast near the entrance?
- Is there sudden hand or body motion near a jewelry case?
- Is the motion abnormal compared to that person's recent baseline?
- Did the movement persist for multiple frames?
- Is the person wearing a face covering?
- Is the wrist or hand near or inside a protected display case zone?

The result is a more realistic surveillance prototype that treats motion as one risk signal, not as proof of criminal behavior.

---

## Key Features

### Person Detection and Tracking

The system uses YOLO-based person detection to identify people in the camera feed. Detected people are tracked across frames so the system can measure movement over time.

This allows the system to understand:

- How many people are in the store
- Which person is near a jewelry case
- Whether a person is moving quickly
- Whether a person remains near a sensitive area for too long
- Whether the same person is repeatedly involved in risky movement

---

### Face Covering Detection

The system includes a trained face-covering detection model that classifies detected faces as masked or unmasked.

Face covering by itself does not automatically mean criminal behavior. However, when combined with other risk signals such as fast entrance movement, loitering, or hand motion near a jewelry case, it increases the overall risk score.

Example risk combination:

```text
Face covering detected
+ Person near jewelry case
+ Sudden hand/body movement near case
= Higher risk event
```

---

### Polygon-Based Security Zones

The system supports user-defined polygon zones instead of relying only on fixed rectangular areas.

Supported zones include:

- Entrance zones
- Jewelry display case zones
- Protected case boundaries

Polygon zones make the system more realistic because store layouts are rarely perfect rectangles. The system can monitor only the areas that matter instead of treating the entire camera frame as equally important.

---

### Fast Entrance Movement Detection

A common mistake in surveillance projects is to use raw pixel speed, such as `pixels per second`. This is unreliable because a person close to the camera appears to move faster than a person farther away.

JewelGuard AI uses normalized body movement instead.

Instead of only checking:

```text
How many pixels did the person move?
```

The system checks:

```text
How fast did the person move relative to their own body height?
```

The entrance motion system uses:

- Normalized body speed
- Acceleration
- Spike ratio compared to that person's recent movement
- Multi-frame confirmation
- Entrance-zone context

This helps detect someone rushing into or out of the entrance while reducing false alarms from normal walking.

---

### Jewelry Case Hand/Body Motion Detection

Near jewelry cases, whole-body speed is not enough. A person may stand mostly still but move their hand quickly toward the case.

The system detects local motion only where the person's expanded bounding box overlaps with the jewelry case zone.

This means the system does not treat all motion as suspicious. It specifically checks for sudden movement in sensitive areas.

The case motion system uses:

- Person-case overlap region
- Frame difference motion detection
- Recent motion baseline
- Motion spike ratio
- Minimum dwell time near the case
- Multi-frame confirmation

This helps separate normal customer movement from abnormal hand/body motion near a display case.

---

### Wrist and Case Boundary Detection

The system can use pose/keypoint information to detect whether a person's wrist or hand is near the display case boundary.

This is important because jewelry theft often involves reaching toward or into the display case area.

The system can detect:

- Wrist near the case boundary
- Wrist inside the protected case zone
- Hand motion combined with face covering
- Hand motion combined with dwell time

---

### Risk Scoring System

The system calculates a live risk score instead of making a simple yes/no decision.

Risk is based on multiple factors:

- Face covering detected
- Person near jewelry case
- Fast movement near entrance
- Wrist near display case
- Wrist inside protected case zone
- Sudden hand/body movement near case
- Loitering near case
- Repeated confirmed fast motion

Example risk levels:

```text
LOW     - Normal customer behavior
MEDIUM  - Suspicious context detected
HIGH    - Strong suspicious behavior detected
```

The system is designed so that normal movement alone does not trigger a critical alert.

---

## Why This Project Is Different From Basic Motion Detection

Many OpenCV tutorials use frame difference or background subtraction to draw contours around moving objects.

Basic motion detection works like this:

```text
Pixels changed -> Draw contour -> Motion detected
```

That is useful, but it does not tell whether the movement is suspicious.

Normal events that can trigger basic motion detection include:

- A customer walking
- An employee moving
- Lighting changes
- Reflections on glass
- Camera noise
- A person turning around
- Someone moving their hands while talking

JewelGuard AI uses motion as only one signal. It combines motion with person tracking, zones, baseline comparison, dwell time, and risk logic.

The improved logic works like this:

```text
Person detected
+ Person tracked across frames
+ Person inside sensitive zone
+ Movement is abnormal compared to baseline
+ Movement persists across multiple frames
+ Risk context exists
= Security event
```

This makes the system more realistic for a jewelry store environment.

---

## Tech Stack

### Backend

- Python
- OpenCV
- NumPy
- YOLOv8
- Ultralytics
- TensorFlow / Keras
- FastAPI or backend API logic
- MediaPipe or pose/keypoint logic

### Frontend

- React
- Vite
- JavaScript
- Dashboard UI for live security status

### Computer Vision Techniques

- Person detection
- Person tracking
- Face-covering classification
- Motion detection
- Polygon zone monitoring
- Pose/keypoint-based wrist detection
- Risk scoring

---

## Project Structure

```text
JewelGuard-AI/
|
|-- backend/
|   |-- backend/
|   |   |-- vision_engine.py
|   |   |-- main.py
|   |   |-- other backend files
|   |
|   |-- requirements.txt
|
|-- frontend/
|   |-- jewelguard-dashboard/
|       |-- src/
|       |   |-- App.jsx
|       |   |-- components/
|       |   |-- other frontend files
|       |
|       |-- package.json
|       |-- vite.config.js
|
|-- vision/
|   |-- training files
|   |-- model files
|   |-- testing scripts
|
|-- data/
|   |-- sample videos
|   |-- raw data
|   |-- processed data
|
|-- README.md
```

---

## Detection Pipeline

### Step 1: Capture Video

The system reads frames from a webcam, CCTV video, or sample video file.

```text
Camera / video input -> OpenCV frame
```

---

### Step 2: Detect People

YOLO detects people in the frame.

```text
Frame -> YOLO person detection -> Person bounding boxes
```

---

### Step 3: Track People

Detected people are tracked across frames so the system can calculate motion over time.

```text
Person box frame 1
Person box frame 2
Person box frame 3
-> movement history
```

---

### Step 4: Check Security Zones

The system checks whether each person is inside or near:

- Entrance zone
- Jewelry case zone
- Protected case boundary

---

### Step 5: Analyze Motion

The system performs two different types of motion analysis.

#### Entrance Motion

Used for detecting fast movement toward or through the entrance.

Signals:

```text
Normalized body speed
Acceleration
Spike ratio
Multi-frame confirmation
```

#### Case Motion

Used for detecting sudden hand/body movement near the jewelry case.

Signals:

```text
Person-case overlap motion
Motion baseline
Motion spike ratio
Dwell time
Multi-frame confirmation
```

---

### Step 6: Analyze Face Covering

The face-covering model checks whether the person appears masked or unmasked.

---

### Step 7: Analyze Wrist/Hand Position

The system checks whether a wrist or hand is near or inside the protected case zone.

---

### Step 8: Calculate Risk

All signals are combined into a risk score.

Example:

```text
Person near case = +15
Wrist near case = +25
Sudden case motion = +35
Face covering = +40
```

The final score determines whether the event is normal, medium risk, or high risk.

---

## Motion Detection Logic

### Entrance Fast Movement

The entrance system does not use raw pixels per second alone.

It uses:

```text
normalized_speed = pixel_speed / person_height
```

This makes speed more reliable across camera distance.

The system confirms fast entrance movement only when:

```text
normalized speed is high
AND acceleration is high
AND movement is a spike compared to recent baseline
AND condition lasts for multiple frames
AND person is inside entrance zone
```

---

### Jewelry Case Fast Motion

The case system does not treat all motion as suspicious.

It checks only the area where the person overlaps with the jewelry case zone.

The system confirms suspicious case movement only when:

```text
person is near jewelry case
AND person has been near the case for enough time
AND motion occurs in the person-case overlap area
AND motion is above recent baseline
AND motion lasts for multiple frames
```

This reduces false alerts from normal walking.

---

## Important Configuration Values

Important thresholds can be tuned inside:

```text
backend/backend/vision_engine.py
```

Example constants:

```python
FAST_BODY_SPEED = 1.25
FAST_BODY_ACCELERATION = 0.55
BODY_SPIKE_RATIO = 2.0
BODY_CONFIRM_FRAMES = 3

CASE_MOTION_MIN = 0.08
CASE_SPIKE_RATIO = 2.5
CASE_CONFIRM_FRAMES = 3
MIN_CASE_DWELL_SECONDS = 1.0
```

If normal movement is being flagged near the case, increase:

```python
CASE_MOTION_MIN = 0.10
CASE_SPIKE_RATIO = 3.0
```

If fast hand movement is being missed, lower:

```python
CASE_MOTION_MIN = 0.06
```

---

## Example Alert Reasons

The dashboard can show reasons such as:

```text
Reduced identity visibility / face covering detected
Fast approach detected near entrance
Person near jewelry display case
Wrist/hand near display case boundary
Wrist/hand entered protected case zone
Sudden hand/body movement near jewelry case
Loitering near jewelry display
Face covering while near jewelry case
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

---

### 2. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

If you are using a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3. Install frontend dependencies

```bash
cd frontend/jewelguard-dashboard
npm install
```

---

## Running the Project

### Start the backend

```bash
cd backend
python -m backend.main
```

Or, depending on your backend setup:

```bash
uvicorn backend.main:app --reload
```

---

### Start the frontend

```bash
cd frontend/jewelguard-dashboard
npm run dev
```

Then open the local Vite URL shown in the terminal, usually:

```text
http://localhost:5173
```

---

## How to Use

1. Start the backend server.
2. Start the frontend dashboard.
3. Open the dashboard in the browser.
4. Select or define security zones such as entrance and jewelry case zones.
5. Start the video stream or webcam.
6. Watch the live risk score, alert type, and alert reasons.
7. Tune motion thresholds based on your camera angle and sample footage.

---

## Limitations

This project is a prototype and should not be used as a final real-world security system without further testing.

Current limitations include:

- Lighting changes may still affect motion detection
- Reflections from glass display cases can create noise
- Camera angle strongly affects visibility of hands and wrists
- Face-covering detection may fail if the face is turned away
- Pose/keypoint detection may fail during occlusion
- Real-world deployment would require more camera views and testing
- The system identifies risk signals, not guilt or criminal intent

---

## Future Improvements

Planned improvements include:

- Better hand tracking near jewelry cases
- Wrist velocity and direction detection
- Multi-camera support
- Staff recognition or employee safe zones
- Re-identification across cameras
- More advanced anomaly detection
- Alert logging with timestamps
- Email or SMS alert integration
- Event replay clips
- Admin dashboard for zone editing
- Better model training with jewelry-store-specific footage

---

## Why This Project Is Valuable

This project demonstrates practical computer vision engineering beyond a basic tutorial. It combines object detection, tracking, motion analysis, zone monitoring, risk scoring, and a live web dashboard.

It shows knowledge of:

- OpenCV video processing
- YOLO object detection
- Real-time AI inference
- Frontend/backend integration
- Security-focused system design
- False-positive reduction
- Context-aware computer vision

The key idea is that motion alone is not suspicious. Suspicious behavior is detected by combining motion with location, speed, timing, hand position, and identity visibility.

---

## GitHub Repository Description

AI-powered jewelry store surveillance prototype using OpenCV, YOLO, face-covering detection, polygon security zones, fast-motion analysis, hand/case proximity detection, and a React dashboard for real-time risk alerts.

---

## Author

Developed by Muneeb Asif as a computer vision and AI security project.

---

## Disclaimer

This project is for educational and portfolio purposes. It is not a replacement for professional security systems, law enforcement judgment, or human monitoring. The system identifies risk signals, not guilt or criminal intent.
