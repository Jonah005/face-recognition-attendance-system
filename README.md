# Face Recognition Attendance System

A local computer vision attendance system built with **Flask** and **OpenCV**.

The application supports separate teacher and student login flows. Students first enter an attendance code and then verify their identity through webcam-based facial recognition. When the expected face is successfully recognized, the student's attendance status is updated to **Present**.

## Features

- Teacher and student login flows
- Attendance validation code
- Webcam-based face capture
- LBPH face recognition using OpenCV
- Haar Cascade face detection
- Student-to-face-label mapping
- In-memory attendance tracking
- Flask web interface with HTML templates
- Environment-based configuration for credentials and application settings

## How It Works

```text
Student Login
     ↓
Attendance Code Validation
     ↓
Webcam Face Recognition
     ↓
Expected Student Face Check
     ↓
Attendance Marked Present
```

The application uses OpenCV's **LBPHFaceRecognizer** for local face recognition and a Haar Cascade classifier for face detection.

## Project Structure

```text
face-recognition-attendance-system/
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── login_student.html
│   ├── login_teacher.html
│   ├── student_attendance.html
│   ├── student_face_recognition.html
│   ├── students_waiting.html
│   ├── teachers_waiting.html
│   └── test.html
├── main.py
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

Local runtime files such as the face dataset, saved labels, and trained face-recognition model are intentionally excluded from the public repository.

## Requirements

- Python 3.10+ recommended
- A working webcam
- OpenCV with the contrib modules enabled

## Installation

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder values with your local configuration.

## Environment Configuration

The application reads its settings from `.env`.

Example:

```env
FLASK_SECRET_KEY=change-this-secret-key
FLASK_DEBUG=false

TEACHER_USERNAME=teacher1
TEACHER_PASSWORD=change-me

STUDENT1_USERNAME=student1
STUDENT1_PASSWORD=change-me
STUDENT1_FACE_LABEL=person1

STUDENT2_USERNAME=student2
STUDENT2_PASSWORD=change-me
STUDENT2_FACE_LABEL=person2

ATTENDANCE_CODE=12345
```

Do not commit your real `.env` file.

## Face Data and Model Files

The application uses the following local files:

```text
face_dataset/
face_labels.npy
face_model.yml
```

These files are excluded from Git because they may contain or represent biometric data.

### `face_dataset/`

Contains locally captured face images used for training.

### `face_labels.npy`

Stores the mapping between face labels and numeric IDs.

### `face_model.yml`

Stores the trained LBPH face-recognition model.

## Running the Application

Start the Flask application:

```powershell
python main.py
```

Then open the local Flask address shown in the terminal.

The default Flask development server normally runs at:

```text
http://127.0.0.1:5000
```

## Attendance Flow

### Teacher

A teacher logs in through the web interface and can view the current attendance state and the configured numerical attendance code.

### Student

A student:

1. Logs in with their configured credentials.
2. Enters the numerical attendance code.
3. Opens the face-recognition step.
4. Allows the application to use the webcam.
5. Is marked **Present** when the expected face is recognized.

## Face Recognition

The system uses:

- OpenCV Haar Cascade for face detection
- OpenCV LBPH for facial recognition
- a configurable recognition threshold
- a local webcam feed for verification

The saved facial label for each student is configured through environment variables such as:

```env
STUDENT1_FACE_LABEL=person1
STUDENT2_FACE_LABEL=person2
```

## Privacy

Biometric data should be handled carefully.

This public repository does **not** include:

- captured face images
- trained face-recognition models
- saved face-label files
- real passwords
- Flask secrets
- private environment files

## Tech Stack

- Python
- Flask
- OpenCV
- NumPy
- python-dotenv
- HTML

## Notes

This project is intended as a local demonstration of computer-vision-based attendance automation.

Attendance is stored in memory while the Flask application is running, so it is not intended as a production attendance-management system without additional persistence, authentication, security, and privacy controls.
