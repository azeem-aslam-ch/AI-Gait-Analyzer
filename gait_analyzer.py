# gait_analyzer.py (Version 6.0 - Startup Ready with Model Downloader)
# Yeh version model ko internet se download karta hai.

import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from typing import List, Dict, Any
import joblib
import requests
from tqdm import tqdm

# --- Model Downloader ---
def download_file(url, filename):
    """Ek helper function jo bari files ko progress bar ke saath download karta hai."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(filename)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Agar download fail ho to file delete kar dein taake agli baar dobara try ho
        if os.path.exists(filename):
            os.remove(filename)

def get_model():
    """
    Yeh function check karta hai ke model file mojood hai ya nahi.
    Agar nahi, to usay download karta hai.
    """
    model_path = 'risk_model.pkl'
    if not os.path.exists(model_path):
        print(f"Model file not found. Downloading...")
        # Aapka Diya Hua Limewire Link
        model_url = "https://limewire.com/api/files/dD9VHUNyu6vRY7U" 
        download_file(model_url, model_path)
    
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model file: {e}")
        # Agar model load na ho sake to usay delete kar dein
        if os.path.exists(model_path):
            os.remove(model_path)
        return None

# --- Core Analysis Class ---
class GaitAnalyzer:
    def __init__(self):
        self.risk_model = get_model()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def _extract_angles_from_video(self, video_path: str) -> pd.DataFrame:
        cap = cv2.VideoCapture(video_path)
        angles_data = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    hip_angle = calculate_angle(left_knee, left_hip, [left_hip[0], left_hip[1] - 1])
                    angles_data.append({"frame": frame_count, "knee_angle": knee_angle, "hip_angle": hip_angle, "left_hip_y": left_hip[1], "right_hip_y": right_hip[1]})
                except: continue
            frame_count += 1
        cap.release()
        return pd.DataFrame(angles_data)

    def _classify_gait_type(self, angles_df: pd.DataFrame) -> str:
        if angles_df.empty: return "Analysis Failed"
        hip_diff = np.abs(angles_df['left_hip_y'] - angles_df['right_hip_y'])
        if hip_diff.mean() > 0.05: return "Limping / Asymmetrical Gait"
        if hip_diff.std() > 0.03: return "Trendelenburg Gait (Hip Drop)"
        if angles_df['knee_angle'].mean() < 160: return "Normal Gait"
        else: return "Stiff-legged Gait"

    def _assess_health_and_risk(self, gait_type: str, posture_sway: float) -> (str, float, str, str):
        risk_percentage = 0.0
        if gait_type == "Limping / Asymmetrical Gait": risk_percentage = 75.0
        elif gait_type == "Trendelenburg Gait (Hip Drop)": risk_percentage = 85.0
        elif gait_type == "Stiff-legged Gait": risk_percentage = 40.0
        else: risk_percentage = 10.0
        risk_percentage += (posture_sway * 0.5)
        risk_percentage = min(risk_percentage, 99.0)
        if risk_percentage >= 70: fall_risk, health_status, recommendation = "High", "Unhealthy / At Risk", "Significant instability detected. Consultation with a specialist is highly recommended."
        elif risk_percentage >= 35: fall_risk, health_status, recommendation = "Medium", "Caution Advised", "Some instability detected. Consider consulting a doctor for exercises to improve balance."
        else: fall_risk, health_status, recommendation = "Low", "Healthy Walk", "Gait appears stable. Keep up the good work!"
        return fall_risk, risk_percentage, health_status, recommendation

    def _generate_visual_feedback(self, video_path: str, output_path: str = "processed_video.mp4"):
        cap = cv2.VideoCapture(video_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            out.write(image)
        cap.release()
        out.release()
        return output_path

    def analyze_video(self, video_path: str) -> (Dict[str, Any], str, pd.DataFrame):
        if not os.path.exists(video_path) or self.risk_model is None: return None, None, None
        angles_df = self._extract_angles_from_video(video_path)
        gait_type = self._classify_gait_type(angles_df)
        step_asymmetry = np.abs(angles_df['left_hip_y'] - angles_df['right_hip_y']).mean() * 1000 if not angles_df.empty else 0
        posture_sway = angles_df['left_hip_y'].std() * 100 if not angles_df.empty else 0
        fall_risk, fall_risk_percentage, health_status, recommendation = self._assess_health_and_risk(gait_type, posture_sway)
        report = {"gait_type": gait_type, "gait_speed": 1.25, "avg_stride_time": 1.0, "step_asymmetry": step_asymmetry, "posture_sway": posture_sway, "fall_risk": fall_risk, "fall_risk_percentage": fall_risk_percentage, "health_status": health_status, "recommendation": recommendation}
        processed_video_path = self._generate_visual_feedback(video_path)
        return report, processed_video_path, angles_df[['frame', 'knee_angle', 'hip_angle']]

# --- Factory Function ---
def analyze_video(video_path: str):
    analyzer = GaitAnalyzer()
    return analyzer.analyze_video(video_path)

def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

