import os
import cv2
import numpy as np
import time
from flask import Flask, render_template, request, redirect, url_for, flash, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key

# Path to save face images and model
dataset_path = os.path.join(os.path.dirname(__file__), 'face_dataset')
model_path = os.path.join(os.path.dirname(__file__), 'face_model.yml')

# Initialize the face recognizer and face detector
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Predefined teacher and student credentials
teachers = {'teacher1': 'password1'}
students = {'student1': 'password1', 'student2': 'password2'}

# Map students to person labels
student_to_person = {'student1': 'person1', 'student2': 'person2'}

# Initialize attendance dictionary
attendance = {student: 'Absent' for student in students}

labels = {}
current_id = 0

# Load previously saved labels if available
label_path = os.path.join(os.path.dirname(__file__), 'face_labels.npy')
if os.path.exists(label_path):
    labels = np.load(label_path, allow_pickle=True).item()
    current_id = max(labels.values()) + 1

# Numerical code for validation
numerical_code = int  # Set a numerical code for students to enter

# Function to add a new face to the known faces
def add_new_face(name):
    global current_id
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            file_name = f"{dataset_path}/{name}_{count}.jpg"
            cv2.imwrite(file_name, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Face {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Adding New Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break

    cap.release()
    cv2.destroyAllWindows()
    if name not in labels:
        labels[name] = current_id
        current_id += 1
        np.save(label_path, labels)

# Function to train the face recognizer
def train_model():
    face_samples = []
    ids = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('jpg'):
                path = os.path.join(root, file)
                label = file.split('_')[0]
                id_ = labels[label]
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                face_samples.append(img)
                ids.append(id_)

    face_recognizer.train(face_samples, np.array(ids))
    face_recognizer.save(model_path)
    print("Model trained and saved.")

# Function to recognize faces in the video stream
def recognize_faces(student_name):
    face_recognizer.read(model_path)
    inverse_labels = {v: k for k, v in labels.items()}

    cap = cv2.VideoCapture(0)
    recognized = False
    recognized_name = None
    start_time = time.time()

    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            id_, confidence = face_recognizer.predict(face_img)
            name = inverse_labels.get(id_, "Unknown")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({round(100 - confidence)}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if name == student_to_person[student_name] and confidence <= 40:  # Change the threshold here (100 - 60)
                recognized = True
                recognized_name = name
                break

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or recognized:
            break

    cap.release()
    cv2.destroyAllWindows()

    if recognized:
        attendance[student_name] = 'Present'

    return recognized, recognized_name


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        if role == 'teacher' and teachers.get(username) == password:
            session['username'] = username
            session['role'] = 'teacher'
            return redirect(url_for('teachers_waiting'))
        elif role == 'student' and students.get(username) == password:
            session['username'] = username
            session['role'] = 'student'
            return redirect(url_for('students_waiting'))
        else:
            flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/teachers_waiting')
def teachers_waiting():
    if 'username' not in session or session['role'] != 'teacher':
        return redirect(url_for('login'))

    return render_template('teachers_waiting.html', attendance=attendance, numerical_code=numerical_code)

@app.route('/students_waiting', methods=['GET', 'POST'])
def students_waiting():
    if 'username' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    student_name = session['username']
    if request.method == 'POST':
        entered_code = request.form['numerical_code']
        if int(entered_code) == numerical_code:
            return redirect(url_for('student_face_recognition'))
        else:
            flash('Incorrect numerical code. Please try again.')
    return render_template('students_waiting.html')

@app.route('/student_face_recognition', methods=['GET', 'POST'])
def student_face_recognition():
    if 'username' not in session or session['role'] != 'student':
        return redirect(url_for('login'))
    student_name = session['username']
    if request.method == 'POST':
        recognized, recognized_name = recognize_faces(student_name)
        if recognized:
            if recognized_name == student_to_person[student_name]:
                attendance[student_name] = 'Present'
                flash('Attendance confirmed.')
            else:
                flash('Wrong person. Please try again.')
            return redirect(url_for('students_waiting'))
        else:
            flash('Face not recognized. Please try again.')
    return render_template('student_face_recognition.html', student_name=student_name, attendance=attendance)


if __name__ == '__main__':
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    app.run(debug=True)
