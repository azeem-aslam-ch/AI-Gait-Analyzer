# gait_analyzer.py (Version 5.4 - Typo Fix)

import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from typing import List, Dict, Any

# --- Helper Functions ---

def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """3 points se angle calculate karta hai."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_landmark_coords(landmarks, landmark_enum) -> List[float]:
    """Ek specific landmark ke (x, y) coordinates hasil karta hai."""
    return [landmarks[landmark_enum.value].x, landmarks[landmark_enum.value].y]

# --- Core Analysis Class ---

class GaitAnalyzer:
    """
    Ek class jo video se gait analysis karti hai.
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def _extract_angles_from_video(self, video_path: str) -> pd.DataFrame:
        """Video se har frame par angles nikalta hai."""
        cap = cv2.VideoCapture(video_path)
        angles_data = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    left_hip = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
                    left_knee = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE)
                    left_ankle = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE)
                    right_hip = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)

                    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    hip_angle = calculate_angle(left_knee, left_hip, [left_hip[0], left_hip[1] - 1])

                    angles_data.append({
                        "frame": frame_count,
                        "knee_angle": knee_angle,
                        "hip_angle": hip_angle,
                        "left_hip_y": left_hip[1],
                        "right_hip_y": right_hip[1]
                    })
                except:
                    continue
            frame_count += 1
        
        cap.release()
        return pd.DataFrame(angles_data)

    def _classify_gait_type(self, angles_df: pd.DataFrame) -> str:
        """Angles ke data par gait type classify karta hai."""
        if angles_df.empty:
            return "Analysis Failed"
            
        hip_diff = np.abs(angles_df['left_hip_y'] - angles_df['right_hip_y'])
        if hip_diff.mean() > 0.05:
             return "Limping / Asymmetrical Gait"
        if hip_diff.std() > 0.03:
            return "Trendelenburg Gait (Hip Drop)"
        if angles_df['knee_angle'].mean() < 160:
            return "Normal Gait"
        else:
            return "Stiff-legged Gait"

    def _assess_health_and_risk(self, gait_type: str, posture_sway: float) -> (str, float, str, str):
        """Gait type aur metrics ki buniyad par health status aur fall risk predict karta hai."""
        risk_percentage = 0.0

        if gait_type == "Limping / Asymmetrical Gait":
            risk_percentage = 75.0
        elif gait_type == "Trendelenburg Gait (Hip Drop)":
            risk_percentage = 85.0
        elif gait_type == "Stiff-legged Gait":
            risk_percentage = 40.0
        else: # Normal Gait
            risk_percentage = 10.0

        # Posture sway ke hisab se risk percentage ko adjust karein
        risk_percentage += (posture_sway * 0.5)
        risk_percentage = min(risk_percentage, 99.0) # Risk ko 99% par cap karein

        # Risk level aur health status ko final percentage ki buniyad par set karein
        if risk_percentage >= 70:
            fall_risk = "High"
            health_status = "Unhealthy / At Risk"
            recommendation = "Significant instability detected. Consultation with a specialist is highly recommended."
        elif risk_percentage >= 35:
            fall_risk = "Medium"
            health_status = "Caution Advised"
            recommendation = "Some instability detected. Consider consulting a doctor for exercises to improve balance."
        else:
            fall_risk = "Low"
            health_status = "Healthy Walk"
            recommendation = "Gait appears stable. Keep up the good work!"
            
        return fall_risk, risk_percentage, health_status, recommendation

    def _generate_visual_feedback(self, video_path: str, output_path: str = "processed_video.mp4"):
        """Video par skeleton draw karta hai."""
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
            out.write(image)
        
        cap.release()
        out.release()
        return output_path

    def analyze_video(self, video_path: str) -> (Dict[str, Any], str, pd.DataFrame):
        if not os.path.exists(video_path):
            return None, None, None

        angles_df = self._extract_angles_from_video(video_path)
        gait_type = self._classify_gait_type(angles_df)

        gait_speed = 1.25 # Dummy value
        avg_stride_time = 1.0 # Dummy value
        step_asymmetry = np.abs(angles_df['left_hip_y'] - angles_df['right_hip_y']).mean() * 1000 if not angles_df.empty else 0
        posture_sway = angles_df['left_hip_y'].std() * 100 if not angles_df.empty else 0

        fall_risk, fall_risk_percentage, health_status, recommendation = self._assess_health_and_risk(gait_type, posture_sway)

        report = {
            "gait_type": gait_type,
            "gait_speed": gait_speed,
            "avg_stride_time": avg_stride_time,
            "step_asymmetry": step_asymmetry,
            "posture_sway": posture_sway,
            "fall_risk": fall_risk,
            "fall_risk_percentage": fall_risk_percentage,
            "health_status": health_status,
            "recommendation": recommendation
        }

        processed_video_path = self._generate_visual_feedback(video_path)

        return report, processed_video_path, angles_df[['frame', 'knee_angle', 'hip_angle']]

# --- Factory Function ---
def analyze_video(video_path: str):
    analyzer = GaitAnalyzer()
    return analyzer.analyze_video(video_path)





