import cv2
import numpy as np
import re
import os
import time
import csv
import json
from tkinter import *
from tkinter import messagebox
from tkinter import Tk, Label, Entry, Button, PhotoImage
from PIL import Image, ImageTk

def resize_image(image_path, width, height):
    original_image = Image.open(image_path)
    resized_image = original_image.resize((width, height))
    return ImageTk.PhotoImage(resized_image)

# Function to create a Tkinter GUI
def create_gui():
    # Define Tkinter window
    root = Tk()
    root.title("Face Recognition Attendance System")
    root.geometry('500x500')
    
    target_width = 500
    target_height = 300
    image_path = r"C:\Users\karti\OneDrive\Desktop\MiniProject\img.gif" #replace with path in your system
    resized_image = resize_image(image_path, target_width, target_height)
    # Function to handle button click for capturing student images
    def capture_images():
        name = name_entry.get()
        roll = roll_entry.get()
        student_data = get_student_data(name,roll)
        try:
            with open('data.json','r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = []
            
        data.append(student_data)

        with open("data.json", "w") as file:
            json.dump(data, file, indent=2)
        capture(name, roll)
        messagebox.showinfo("Success", "Images captured successfully!")

    # Function to handle button click for marking attendance
    def mark_attendance():
        # Call the markAttendance function from the provided code
        markAttendance()
        #messagebox.showinfo("Success", "Attendance marked successfully!")
    
    # GUI elements
    label_image = Label(root, image=resized_image)
    label_name = Label(root, text="Enter student name:")
    label_roll = Label(root, text="Enter roll number:")
    name_entry = Entry(root)
    roll_entry = Entry(root)

    # Buttons for capturing images and marking attendance
    capture_button = Button(root, text="Capture Images", command=capture_images)
    mark_button = Button(root, text="Mark Attendance", command=mark_attendance)
    
    # Placing GUI elements in the window
    label_image.grid(row=0, column=0, columnspan=2)
    label_name.grid(row=1, column=0)
    name_entry.grid(row=1, column=1)
    label_roll.grid(row=2, column=0)
    roll_entry.grid(row=2, column=1)
    capture_button.grid(row=3, column=0, columnspan=2, pady=10)
    mark_button.grid(row=4, column=0, columnspan=2, pady=10)

    # Run the Tkinter main loop
    root.mainloop()

def get_student_data(name,roll_number):
    return {roll_number: name}

def find_student_name(roll_number,data):
    for student in data:
        if roll_number in student:
            return student[roll_number]
    return None
def get_name_by_roll(label):
    try:
        with open('data.json','r') as file:
            data = json.load(file)
        student_name = find_student_name(label, data)
        if student_name is not None:
            return student_name
        else:
            return None
    except FileNotFoundError:
        print("No Student data found")

# Load the database images
def load_image_from_folder(folder):
    faces = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            label = re.findall(r'\d+',filename)
            if label:
                label = int(label[0])
                faces.append(img)
                labels.append(label)
    return faces, labels
# Capture the Students Images
def capture(name , roll):
    student_details = {}
    # Load module
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')    
    # Access the webcam
    cap = cv2.VideoCapture(0)

    count = 0

    max_images_per_student = 5

    while True:
        ret, frame = cap.read() # read the frames 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert gray scale

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) # to detect faces in the frame or image capture from webcam

        # draw rectangle around the face
        for(x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            cv2.rectangle(frame,(x,y),(x + w , y + h), (0, 255, 0), 3)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if count < max_images_per_student:
                    img_name = f"{roll}_{count}.png"
                    img_path = os.path.join(data_dir, img_name)
                    cv2.imwrite(img_path, roi_gray)
                    student_details[img_name] = name

                    count += 1
                    print(f"Image {count} captured for {name}")

                    if count == max_images_per_student:
                        count = 0
                else:
                    print("Maximum images are already captured for this student")        
    
    # Display the output
        cv2.imshow('capture Images for students', frame)

    # break the loop jab apna kaam ho jaye by pressing 'q'    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# Release the webcam and close the windows (OpenCV)
    roll = int(roll) + 1
    cap.release()
    cv2.destroyAllWindows()
          



def markAttendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    folder_path = "C:\\Users\\karti\\OneDrive\\Desktop\\MiniProject\\student_data" #replace with path in your system
    stored_faces, labels = load_image_from_folder(folder_path)
    # Training the model
    recognizer.train(stored_faces, np.array(labels))

    
    student_details = {}
    for filename in os.listdir(data_dir):
        label = re.findall(r'\d+', filename)
        if label:
            label = int(label[0])
            student_details[filename] = f"Student_{label}"

    # Capture a new image to detect
    capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    confirm = 0

    # Open camera window without capturing any frames
    cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Recognition', 800, 600)

    # Delay for 3 seconds
    print("Get ready, the camera will start in 3 seconds...")
    time.sleep(3)

    # Create or open the CSV file to write attendance
    csv_file_path = 'attendance.csv'
    file_exists = os.path.exists(csv_file_path)
    with open(csv_file_path,mode='a',newline='') as csvfile:
        fieldnames = ['Roll No.', 'Name', 'Timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Image not read properly")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                label, confidence = recognizer.predict(face)
                threshold = 70
                if confidence < threshold:
                    name = get_name_by_roll(str(label))
                    confirm = 3
                    # Write attendance to CSV
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
                    writer.writerow({'Roll No.': label, 'Name': str(name), 'Timestamp': timestamp})
                    break

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if confirm == 3 or count == 50:
                break

            cv2.imshow('Face Recognition', frame)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
    if not confirm:
        messagebox.showinfo("Student not in the Database!")
    else:
        messagebox.showinfo("Success", "Attendance marked successfully!")

if __name__ == "__main__":
        data_dir = 'student_data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        create_gui()