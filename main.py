import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
)


# ---------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "development-secret-key-change-me"
)


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(BASE_DIR, "face_dataset")
model_path = os.path.join(BASE_DIR, "face_model.yml")
label_path = os.path.join(BASE_DIR, "face_labels.npy")


# ---------------------------------------------------------
# Face recognition configuration
# ---------------------------------------------------------

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------------------------------------------------
# User configuration
# ---------------------------------------------------------

teacher_username = os.getenv("TEACHER_USERNAME", "teacher1")
teacher_password = os.getenv("TEACHER_PASSWORD", "change-me")

student1_username = os.getenv("STUDENT1_USERNAME", "student1")
student1_password = os.getenv("STUDENT1_PASSWORD", "change-me")

student2_username = os.getenv("STUDENT2_USERNAME", "student2")
student2_password = os.getenv("STUDENT2_PASSWORD", "change-me")


teachers = {
    teacher_username: teacher_password
}

students = {
    student1_username: student1_password,
    student2_username: student2_password,
}


# ---------------------------------------------------------
# Student → facial-recognition label mapping
# ---------------------------------------------------------

student_to_person = {
    student1_username: os.getenv(
        "STUDENT1_FACE_LABEL",
        "person1"
    ),
    student2_username: os.getenv(
        "STUDENT2_FACE_LABEL",
        "person2"
    ),
}


# ---------------------------------------------------------
# Attendance
# ---------------------------------------------------------

attendance = {
    student: "Absent"
    for student in students
}


# ---------------------------------------------------------
# Face labels
# ---------------------------------------------------------

labels = {}
current_id = 0

if os.path.exists(label_path):
    try:
        labels = np.load(
            label_path,
            allow_pickle=True
        ).item()

        if labels:
            current_id = max(labels.values()) + 1

    except Exception as error:
        print(
            f"Could not load face labels: {error}"
        )


# ---------------------------------------------------------
# Attendance validation code
# ---------------------------------------------------------

try:
    numerical_code = int(
        os.getenv(
            "ATTENDANCE_CODE",
            "12345"
        )
    )
except ValueError:
    numerical_code = 12345


# ---------------------------------------------------------
# Add a new face
# ---------------------------------------------------------

def add_new_face(name):
    """
    Capture face images from the webcam and store them
    in the local face dataset.
    """

    global current_id

    os.makedirs(
        dataset_path,
        exist_ok=True
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(
            "Unable to access webcam."
        )
        return False

    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print(
                "Unable to read frame from webcam."
            )
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for x, y, w, h in faces:

            count += 1

            face_img = gray[
                y:y + h,
                x:x + w
            ]

            file_name = os.path.join(
                dataset_path,
                f"{name}_{count}.jpg"
            )

            cv2.imwrite(
                file_name,
                face_img
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Face {count}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        cv2.imshow(
            "Adding New Face",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or count >= 20:
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        return False

    if name not in labels:

        labels[name] = current_id

        current_id += 1

        np.save(
            label_path,
            labels
        )

    return True


# ---------------------------------------------------------
# Train face-recognition model
# ---------------------------------------------------------

def train_model():
    """
    Train the LBPH face recognizer using images stored
    in the local face_dataset directory.
    """

    if not os.path.exists(dataset_path):
        print(
            "Face dataset directory does not exist."
        )
        return False

    face_samples = []
    ids = []

    for root, _, files in os.walk(
        dataset_path
    ):

        for file in files:

            if not file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                )
            ):
                continue

            path = os.path.join(
                root,
                file
            )

            label = file.split("_")[0]

            if label not in labels:
                continue

            image = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            face_samples.append(
                image
            )

            ids.append(
                labels[label]
            )

    if not face_samples:
        print(
            "No valid face samples were found."
        )
        return False

    face_recognizer.train(
        face_samples,
        np.array(ids)
    )

    face_recognizer.save(
        model_path
    )

    print(
        "Model trained and saved."
    )

    return True


# ---------------------------------------------------------
# Recognize student face
# ---------------------------------------------------------

def recognize_faces(student_name):
    """
    Open the webcam and attempt to verify that the face
    matches the logged-in student's configured face label.
    """

    if not os.path.exists(model_path):
        print(
            "Face-recognition model not found."
        )
        return False, None

    if not labels:
        print(
            "No saved face labels were found."
        )
        return False, None

    if student_name not in student_to_person:
        return False, None

    try:
        face_recognizer.read(
            model_path
        )
    except cv2.error as error:

        print(
            f"Unable to load face model: {error}"
        )

        return False, None

    inverse_labels = {
        value: key
        for key, value in labels.items()
    }

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(
            "Unable to access webcam."
        )
        return False, None

    recognized = False
    recognized_name = None

    start_time = time.time()

    # Allow up to 10 seconds for recognition
    while time.time() - start_time < 10:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for x, y, w, h in faces:

            face_img = gray[
                y:y + h,
                x:x + w
            ]

            try:
                id_, confidence = (
                    face_recognizer.predict(
                        face_img
                    )
                )

            except cv2.error:
                continue

            name = inverse_labels.get(
                id_,
                "Unknown"
            )

            display_confidence = max(
                0,
                round(
                    100 - confidence
                )
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{name} ({display_confidence}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            expected_person = (
                student_to_person[
                    student_name
                ]
            )

            # LBPH returns lower confidence values
            # for better matches.
            if (
                name == expected_person
                and confidence <= 40
            ):

                recognized = True
                recognized_name = name

                break

        cv2.imshow(
            "Face Recognition",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or recognized:
            break

    cap.release()
    cv2.destroyAllWindows()

    if recognized:
        attendance[
            student_name
        ] = "Present"

    return (
        recognized,
        recognized_name
    )


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ---------------------------------------------------------
# Login
# ---------------------------------------------------------

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        )

        password = request.form.get(
            "password",
            ""
        )

        role = request.form.get(
            "role",
            ""
        )

        if (
            role == "teacher"
            and teachers.get(username)
            == password
        ):

            session[
                "username"
            ] = username

            session[
                "role"
            ] = "teacher"

            return redirect(
                url_for(
                    "teachers_waiting"
                )
            )

        elif (
            role == "student"
            and students.get(username)
            == password
        ):

            session[
                "username"
            ] = username

            session[
                "role"
            ] = "student"

            return redirect(
                url_for(
                    "students_waiting"
                )
            )

        flash(
            "Invalid credentials. Please try again."
        )

    return render_template(
        "login.html"
    )


# ---------------------------------------------------------
# Teacher waiting page
# ---------------------------------------------------------

@app.route(
    "/teachers_waiting"
)
def teachers_waiting():

    if (
        "username"
        not in session
        or session.get("role")
        != "teacher"
    ):

        return redirect(
            url_for(
                "login"
            )
        )

    return render_template(
        "teachers_waiting.html",
        attendance=attendance,
        numerical_code=numerical_code,
    )


# ---------------------------------------------------------
# Student waiting page
# ---------------------------------------------------------

@app.route(
    "/students_waiting",
    methods=["GET", "POST"]
)
def students_waiting():

    if (
        "username"
        not in session
        or session.get("role")
        != "student"
    ):

        return redirect(
            url_for(
                "login"
            )
        )

    if request.method == "POST":

        entered_code = request.form.get(
            "numerical_code",
            ""
        )

        try:

            entered_code = int(
                entered_code
            )

        except ValueError:

            flash(
                "Please enter a valid numerical code."
            )

            return render_template(
                "students_waiting.html"
            )

        if entered_code == numerical_code:

            return redirect(
                url_for(
                    "student_face_recognition"
                )
            )

        flash(
            "Incorrect numerical code. Please try again."
        )

    return render_template(
        "students_waiting.html"
    )


# ---------------------------------------------------------
# Student face-recognition page
# ---------------------------------------------------------

@app.route(
    "/student_face_recognition",
    methods=["GET", "POST"]
)
def student_face_recognition():

    if (
        "username"
        not in session
        or session.get("role")
        != "student"
    ):

        return redirect(
            url_for(
                "login"
            )
        )

    student_name = session[
        "username"
    ]

    if request.method == "POST":

        recognized, recognized_name = (
            recognize_faces(
                student_name
            )
        )

        if recognized:

            expected_person = (
                student_to_person.get(
                    student_name
                )
            )

            if (
                recognized_name
                == expected_person
            ):

                attendance[
                    student_name
                ] = "Present"

                flash(
                    "Attendance confirmed."
                )

            else:

                flash(
                    "Wrong person. Please try again."
                )

        else:

            flash(
                "Face not recognized. Please try again."
            )

        return redirect(
            url_for(
                "students_waiting"
            )
        )

    return render_template(
        "student_face_recognition.html",
        student_name=student_name,
        attendance=attendance,
    )


# ---------------------------------------------------------
# Run application
# ---------------------------------------------------------

if __name__ == "__main__":

    os.makedirs(
        dataset_path,
        exist_ok=True
    )

    debug_mode = (
        os.getenv(
            "FLASK_DEBUG",
            "false"
        ).lower()
        == "true"
    )

    app.run(
        debug=debug_mode
    )