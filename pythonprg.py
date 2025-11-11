import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk

dataset_path = "faces"
os.makedirs(dataset_path, exist_ok=True)
trainer_path = "trainer.yml"
labels_path = "labels.npy"
attendance_folder = "attendance"
os.makedirs(attendance_folder, exist_ok=True)

root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("950x600")
root.configure(bg="#121212")

accent_color = "#00B7FF"
success_color = "#28A745"
warning_color = "#FFC107"
error_color = "#FF4C4C"
text_color = "#E0E0E0"
font_main = ("Segoe UI", 12)
font_title = ("Segoe UI", 18, "bold")

sidebar = tk.Frame(root, bg="#1E1E1E", width=220)
sidebar.pack(side="left", fill="y")

logo = tk.Label(sidebar, text="Smart Attendance", font=("Segoe UI", 16, "bold"), fg=accent_color, bg="#1E1E1E", pady=20)
logo.pack()

def create_sidebar_button(text, command, color):
    btn = tk.Button(sidebar, text=text, command=command,
                    font=("Segoe UI", 12, "bold"), bg=color, fg="white",
                    relief="flat", activebackground="#333333",
                    activeforeground="white", padx=15, pady=10, width=20)
    btn.pack(pady=10)
    return btn

def capture_faces():
    uid = uid_entry.get().strip()
    name = name_entry.get().strip()
    if not uid or not name:
        messagebox.showwarning("Input Error", "Please enter both UID and Name.")
        return
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Unable to access webcam.")
        return
    sample_count = 0
    user_folder = os.path.join(dataset_path, f"{uid}_{name}")
    os.makedirs(user_folder, exist_ok=True)
    status_label.config(text="Capturing faces... Press 'q' to quit.", fg="yellow")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sample_count += 1
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(user_folder, f"{name}_{sample_count}.jpg"), face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, f"Samples: {sample_count}", (x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or sample_count >= 50:
            break
    cap.release()
    cv2.destroyAllWindows()
    if sample_count >= 10:
        messagebox.showinfo("Success", f"Captured {sample_count} faces for {name}.")
        status_label.config(text=f"Captured {sample_count} faces for {name}.", fg="lightgreen")
    else:
        messagebox.showwarning("Incomplete", "Face capture stopped before enough samples were collected.")

def train_recognizer():
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces = []
        ids = []
        label_dict = {}
        label_id = 0
        for folder_name in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            label_dict[label_id] = folder_name
            for image_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, image_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(img)
                ids.append(label_id)
            label_id += 1
        if not faces:
            messagebox.showwarning("Training Error", "No faces found! Please capture faces first.")
            return
        face_recognizer.train(faces, np.array(ids))
        face_recognizer.save(trainer_path)
        np.save(labels_path, label_dict)
        messagebox.showinfo("Training Complete", "Face recognizer trained successfully.")
        status_label.config(text="Model trained successfully.", fg="lightgreen")
    except Exception as e:
        messagebox.showerror("Error", f"Training failed: {e}")

def mark_attendance():
    try:
        if not os.path.exists(trainer_path) or not os.path.exists(labels_path):
            messagebox.showwarning("Model Error", "Please train the recognizer first.")
            return
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(trainer_path)
        label_dict = np.load(labels_path, allow_pickle=True).item()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access webcam.")
            return
        status_label.config(text="Recognizing faces... Press 'q' to stop.", fg="yellow")
        today = datetime.now().strftime("%Y-%m-%d")
        attendance_file = os.path.join(attendance_folder, f"Attendance_{today}.csv")
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
        else:
            df = pd.DataFrame(columns=["UID", "Name", "Time", "Date"])
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                label_id, confidence = recognizer.predict(face_roi)
                label_name = label_dict.get(label_id, "Unknown")
                confidence_text = f"{round(100 - confidence, 2)}%"
                if confidence < 70:
                    uid, name = label_name.split("_", 1)
                    now = datetime.now()
                    time_str = now.strftime("%H:%M:%S")
                    date_str = now.strftime("%Y-%m-%d")
                    if not ((df["UID"] == uid) & (df["Date"] == date_str)).any():
                        df.loc[len(df)] = [uid, name, time_str, date_str]
                        df.to_csv(attendance_file, index=False)
                    color = (0, 255, 0)
                    text = f"{name} ({confidence_text})"
                else:
                    color = (0, 0, 255)
                    text = "Unknown"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Mark Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Attendance Marked", "Attendance has been recorded successfully.")
        status_label.config(text="Attendance recorded.", fg="lightgreen")
    except Exception as e:
        messagebox.showerror("Error", f"Recognition failed: {e}")

def check_attendance():
    for item in tree.get_children():
        tree.delete(item)
    today = datetime.now().strftime("%Y-%m-%d")
    attendance_file = os.path.join(attendance_folder, f"Attendance_{today}.csv")
    if not os.path.exists(attendance_file):
        messagebox.showinfo("No Data", "No attendance records found for today.")
        return
    df = pd.read_csv(attendance_file)
    for _, row in df.iterrows():
        tree.insert("", "end", values=(row["UID"], row["Name"], row["Time"], row["Date"]))
    status_label.config(text=f"Loaded attendance for {today}.", fg="lightgreen")

btn_capture = create_sidebar_button("Capture Faces", capture_faces, accent_color)
btn_train = create_sidebar_button("Train Recognizer", train_recognizer, success_color)
btn_mark = create_sidebar_button("Mark Attendance", mark_attendance, warning_color)
btn_check = create_sidebar_button("Check Attendance", check_attendance, "#6C63FF")

main_frame = tk.Frame(root, bg="#181818", padx=25, pady=20)
main_frame.pack(side="right", fill="both", expand=True)

title_label = tk.Label(main_frame, text="Face Recognition Attendance System", font=font_title, fg=accent_color, bg="#181818")
title_label.pack(anchor="w", pady=(0, 20))

content_frame = tk.Frame(main_frame, bg="#1F1F1F", padx=20, pady=20)
content_frame.pack(fill="both", expand=True)

input_frame = tk.LabelFrame(content_frame, text="Enter User Details", font=("Segoe UI", 12, "bold"),
                            bg="#242424", fg=accent_color, padx=15, pady=15, relief="groove")
input_frame.pack(fill="x", pady=10)

tk.Label(input_frame, text="UID:", font=font_main, bg="#242424", fg=text_color).grid(row=0, column=0, sticky="w", pady=8)
uid_entry = tk.Entry(input_frame, font=font_main, width=20)
uid_entry.grid(row=0, column=1, padx=10, pady=8)

tk.Label(input_frame, text="Name:", font=font_main, bg="#242424", fg=text_color).grid(row=1, column=0, sticky="w", pady=8)
name_entry = tk.Entry(input_frame, font=font_main, width=30)
name_entry.grid(row=1, column=1, padx=10, pady=8)

history_frame = tk.LabelFrame(content_frame, text="Attendance History", font=("Segoe UI", 12, "bold"),
                              bg="#242424", fg=accent_color, padx=15, pady=15)
history_frame.pack(fill="both", expand=True, pady=10)

tree = ttk.Treeview(history_frame, columns=("UID", "Name", "Time", "Date"), show="headings", height=10)
tree.heading("UID", text="UID")
tree.heading("Name", text="Name")
tree.heading("Time", text="Time")
tree.heading("Date", text="Date")
tree.column("UID", width=100)
tree.column("Name", width=150)
tree.column("Time", width=120)
tree.column("Date", width=120)
tree.pack(fill="both", expand=True)

style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview",
                background="#1E1E1E",
                foreground="white",
                fieldbackground="#1E1E1E",
                rowheight=28,
                font=("Segoe UI", 11))
style.configure("Treeview.Heading", font=("Segoe UI", 12, "bold"), background="#00B7FF", foreground="white")

status_label = tk.Label(root, text="Ready", font=("Segoe UI", 11), fg="lightgreen", bg="#1E1E1E", anchor="w", padx=20)
status_label.pack(side="bottom", fill="x")

root.mainloop()
