import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque, Counter
from typing import Tuple, Optional, Dict, Any, List

class GaitPostureAnalyzer:
    
    HEEL_THRESHOLD = 12
    TOE_THRESHOLD = 12
    MIDFOOT_DIFF_THRESHOLD = 8
    VELOCITY_LOW_THRESHOLD = 0.01
    VELOCITY_MID_THRESHOLD = 0.03
    HIGH_LEG_ANGLE = 150
    LOW_LEG_ANGLE = 120
    MID_LEG_ANGLE = 140
    MAX_ANGLE_LIMIT = 45.0
    DETECTION_BUFFER_SIZE = 10
    STRIKE_HISTORY_SIZE = 10
    POSTURE_HISTORY_SIZE = 5
    DEFAULT_FALLBACK_LEAN = 5.0
    NO_DETECTION_LEAN = 8.0
    
    def __init__(self, detection_confidence: float = 0.1, tracking_confidence: float = 0.1):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False
        )
        
        self.frame_data: List[Dict[str, Any]] = []
        self.current_frame: int = 0
        self.video_fps: float = 30.0
        self.left_strike_history = deque(maxlen=self.STRIKE_HISTORY_SIZE)
        self.right_strike_history = deque(maxlen=self.STRIKE_HISTORY_SIZE)
        self.posture_history = deque(maxlen=self.POSTURE_HISTORY_SIZE)
        self.last_valid_landmarks = None
        self.detection_buffer = []
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pose.close()
    
    def _calculate_leg_angle(self, hip, knee, ankle) -> float:
        try:
            hip_pos = np.array([hip.x, hip.y])
            knee_pos = np.array([knee.x, knee.y])
            ankle_pos = np.array([ankle.x, ankle.y])
            
            v1 = hip_pos - knee_pos
            v2 = ankle_pos - knee_pos
            
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        except (ZeroDivisionError, ValueError):
            return 90.0
    
    def _classify_strike_from_context(self, velocity: float, leg_angle: float, side: str) -> str:
        if velocity < self.VELOCITY_LOW_THRESHOLD:
            if leg_angle > self.HIGH_LEG_ANGLE:
                strike_type = "heel"
            elif leg_angle < self.LOW_LEG_ANGLE:
                strike_type = "forefoot"
            else:
                strike_type = "midfoot"
        elif velocity < self.VELOCITY_MID_THRESHOLD:
            if leg_angle > self.MID_LEG_ANGLE:
                strike_type = "heel"
            else:
                strike_type = "midfoot"
        else:
            strike_type = "midfoot"
        
        history = self.left_strike_history if side == "left" else self.right_strike_history
        history.append(strike_type)
        
        if len(history) >= 3:
            recent_strikes = list(history)[-3:]
            most_common = Counter(recent_strikes).most_common(1)[0][0]
            return most_common
        
        return strike_type
    
    def _estimate_foot_strike_from_velocity(self, landmarks, frame_num: int) -> Tuple[str, str]:
        if frame_num < 3 or len(self.detection_buffer) < 3:
            return "midfoot", "midfoot"
        
        try:
            current_left_ankle = np.array([landmarks[27].x, landmarks[27].y])
            current_right_ankle = np.array([landmarks[28].x, landmarks[28].y])
            
            prev_landmarks = self.detection_buffer[-2] if len(self.detection_buffer) >= 2 else None
            if prev_landmarks is None:
                return "midfoot", "midfoot"
            
            prev_left_ankle = np.array([prev_landmarks[27].x, prev_landmarks[27].y])
            prev_right_ankle = np.array([prev_landmarks[28].x, prev_landmarks[28].y])
            
            left_velocity = np.linalg.norm(current_left_ankle - prev_left_ankle)
            right_velocity = np.linalg.norm(current_right_ankle - prev_right_ankle)
            
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            left_leg_angle = self._calculate_leg_angle(left_hip, left_knee, landmarks[27])
            right_leg_angle = self._calculate_leg_angle(right_hip, right_knee, landmarks[28])
            
            left_strike = self._classify_strike_from_context(left_velocity, left_leg_angle, "left")
            right_strike = self._classify_strike_from_context(right_velocity, right_leg_angle, "right")
            
            return left_strike, right_strike
            
        except (IndexError, AttributeError):
            return "midfoot", "midfoot"
    
    def _determine_strike_type(self, heel_diff: float, toe_diff: float) -> str:
        if heel_diff < self.HEEL_THRESHOLD and heel_diff <= toe_diff:
            return "heel"
        elif toe_diff < self.TOE_THRESHOLD and toe_diff < heel_diff:
            return "forefoot"
        elif abs(heel_diff - toe_diff) < self.MIDFOOT_DIFF_THRESHOLD:
            return "midfoot"
        else:
            return "midfoot"
    
    def _combine_strike_analysis(self, geometric_strike: str, velocity_strike: str, side: str) -> str:
        if geometric_strike == velocity_strike:
            return geometric_strike
        
        history = self.left_strike_history if side == "left" else self.right_strike_history
        
        if len(history) > 0:
            last_strike = list(history)[-1]
            if last_strike == geometric_strike:
                return geometric_strike
            elif last_strike == velocity_strike:
                return velocity_strike
        
        return geometric_strike
    
    def _get_fallback_strikes(self) -> Tuple[str, str]:
        left_fallback = "midfoot"
        right_fallback = "midfoot"
        
        if len(self.left_strike_history) > 0:
            left_fallback = Counter(self.left_strike_history).most_common(1)[0][0]
        
        if len(self.right_strike_history) > 0:
            right_fallback = Counter(self.right_strike_history).most_common(1)[0][0]
        
        return left_fallback, right_fallback
    
    def analyze_foot_strike_robust(self, landmarks, frame_width: int, frame_height: int, frame_num: int) -> Tuple[str, str]:
        try:
            left_heel = landmarks[29]
            left_foot_index = landmarks[31]
            right_heel = landmarks[30]
            right_foot_index = landmarks[32]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            left_heel_y = left_heel.y * frame_height
            left_foot_y = left_foot_index.y * frame_height
            left_ankle_y = left_ankle.y * frame_height
            
            right_heel_y = right_heel.y * frame_height
            right_foot_y = right_foot_index.y * frame_height
            right_ankle_y = right_ankle.y * frame_height
            
            left_heel_diff = abs(left_heel_y - left_ankle_y)
            left_toe_diff = abs(left_foot_y - left_ankle_y)
            right_heel_diff = abs(right_heel_y - right_ankle_y)
            right_toe_diff = abs(right_foot_y - right_ankle_y)
            
            left_strike_type = self._determine_strike_type(left_heel_diff, left_toe_diff)
            right_strike_type = self._determine_strike_type(right_heel_diff, right_toe_diff)
            
            velocity_left, velocity_right = self._estimate_foot_strike_from_velocity(landmarks, frame_num)
            
            left_final = self._combine_strike_analysis(left_strike_type, velocity_left, "left")
            right_final = self._combine_strike_analysis(right_strike_type, velocity_right, "right")
            
            return left_final, right_final
            
        except Exception:
            return self._get_fallback_strikes()
    
    def analyze_posture_advanced(self, landmarks) -> Tuple[float, float]:
        try:
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_ear = landmarks[7]
            right_ear = landmarks[8]
            
            shoulder_center = np.array([(left_shoulder.x + right_shoulder.x) / 2, 
                                     (left_shoulder.y + right_shoulder.y) / 2])
            hip_center = np.array([(left_hip.x + right_hip.x) / 2, 
                                 (left_hip.y + right_hip.y) / 2])
            ear_center = np.array([(left_ear.x + right_ear.x) / 2, 
                                 (left_ear.y + right_ear.y) / 2])
            
            torso_vector = shoulder_center - hip_center
            vertical_vector = np.array([0, -1])
            
            dot_product = np.dot(torso_vector, vertical_vector)
            torso_magnitude = np.linalg.norm(torso_vector)
            
            if torso_magnitude > 0:
                cos_angle = dot_product / torso_magnitude
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                torso_angle = np.degrees(np.arccos(cos_angle))
            else:
                torso_angle = 0.0
            
            head_vector = np.array([nose.x, nose.y]) - ear_center
            head_dot_product = np.dot(head_vector, vertical_vector)
            head_magnitude = np.linalg.norm(head_vector)
            
            if head_magnitude > 0:
                head_cos_angle = head_dot_product / head_magnitude
                head_cos_angle = np.clip(head_cos_angle, -1.0, 1.0)
                head_angle = np.degrees(np.arccos(head_cos_angle))
            else:
                head_angle = 0.0
            
            forward_lean = min(abs(torso_angle), self.MAX_ANGLE_LIMIT)
            head_forward = min(abs(head_angle), self.MAX_ANGLE_LIMIT)
            
            self.posture_history.append((forward_lean, head_forward))
            
            if len(self.posture_history) >= 3:
                recent_leans = [p[0] for p in list(self.posture_history)[-3:]]
                recent_heads = [p[1] for p in list(self.posture_history)[-3:]]
                forward_lean = np.median(recent_leans)
                head_forward = np.median(recent_heads)
            
            return forward_lean, head_forward
            
        except Exception:
            if len(self.posture_history) > 0:
                last_posture = list(self.posture_history)[-1]
                return last_posture[0], last_posture[1]
            return self.DEFAULT_FALLBACK_LEAN, self.DEFAULT_FALLBACK_LEAN
    
    def _interpolate_landmarks(self, current_landmarks, frame_num: int):
        if current_landmarks is not None:
            self.last_valid_landmarks = current_landmarks
            return current_landmarks
        
        if self.last_valid_landmarks is not None and frame_num < 10:
            return self.last_valid_landmarks
        
        return None
    
    def _create_frame_info(self, frame_number: int) -> Dict[str, Any]:
        return {
            'frame_number': frame_number,
            'timestamp': frame_number / self.video_fps,
            'left_foot_strike': 'midfoot',
            'right_foot_strike': 'midfoot',
            'forward_lean': self.DEFAULT_FALLBACK_LEAN,
            'head_forward': self.DEFAULT_FALLBACK_LEAN
        }
    
    def _draw_annotations(self, frame, results, left_strike: str, right_strike: str, 
                         forward_lean: float, head_forward: float, is_estimated: bool = False):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        
        cv2.putText(frame, f"L: {left_strike}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"R: {right_strike}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Pochylenie: {forward_lean:.1f}°", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Głowa: {head_forward:.1f}°", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if is_estimated:
            cv2.putText(frame, "ESTYMACJA", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def process_video(self, video_path: str, output_dir: str = "output") -> str:
        if not video_path.lower().endswith('.mp4'):
            raise ValueError("Obsługiwany jest tylko format MP4")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Nie znaleziono pliku: {video_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Nie można otworzyć filmu: {video_path}")
        
        try:
            self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = os.path.join(output_dir, f"{video_name}_analyzed.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, self.video_fps, (frame_width, frame_height))
            
            if not out.isOpened():
                raise ValueError("Nie można utworzyć pliku wyjściowego")
            
            self._reset_analysis_state()
            
            print(f"Rozpoczynam analizę filmu: {video_name}")
            print(f"Całkowita liczba klatek: {total_frames}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                landmarks = None
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    self.detection_buffer.append(landmarks)
                    if len(self.detection_buffer) > self.DETECTION_BUFFER_SIZE:
                        self.detection_buffer.pop(0)
                
                effective_landmarks = self._interpolate_landmarks(landmarks, self.current_frame)
                frame_info = self._create_frame_info(self.current_frame)
                
                if effective_landmarks:
                    left_strike, right_strike = self.analyze_foot_strike_robust(
                        effective_landmarks, frame_width, frame_height, self.current_frame
                    )
                    forward_lean, head_forward = self.analyze_posture_advanced(effective_landmarks)
                    
                    frame_info.update({
                        'left_foot_strike': left_strike,
                        'right_foot_strike': right_strike,
                        'forward_lean': forward_lean,
                        'head_forward': head_forward
                    })
                    
                    is_estimated = not results.pose_landmarks
                    self._draw_annotations(frame, results, left_strike, right_strike, 
                                         forward_lean, head_forward, is_estimated)
                else:
                    fallback_left, fallback_right = self._get_fallback_strikes()
                    frame_info.update({
                        'left_foot_strike': fallback_left,
                        'right_foot_strike': fallback_right,
                        'forward_lean': self.NO_DETECTION_LEAN,
                        'head_forward': self.NO_DETECTION_LEAN
                    })
                    
                    cv2.putText(frame, "BRAK DETEKCJI - FALLBACK", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                self.frame_data.append(frame_info)
                out.write(frame)
                self.current_frame += 1
                
                if self.current_frame % 30 == 0:
                    progress = (self.current_frame / total_frames) * 100
                    print(f"Przetworzono {self.current_frame}/{total_frames} klatek ({progress:.1f}%)")
            
            print("Analiza zakończona. Przetwarzanie i zapisywanie wyników...")
            
        finally:
            cap.release()
            out.release()
        
        self._post_process_data()
        self._save_analysis_results(video_name, output_dir)
        self._generate_report(video_name, output_dir)
        
        print(f"Wyniki zapisane w folderze: {output_dir}")
        return output_video_path
    
    def _reset_analysis_state(self):
        self.frame_data = []
        self.current_frame = 0
        self.detection_buffer = []
        self.last_valid_landmarks = None
        self.left_strike_history.clear()
        self.right_strike_history.clear()
        self.posture_history.clear()
    
    def _post_process_data(self):
        df = pd.DataFrame(self.frame_data)
        
        for col in ['forward_lean', 'head_forward']:
            values = df[col].values
            if len(values) > 5:
                smoothed = self._smooth_signal(values, window=5)
                df[col] = smoothed
        
        strike_cols = ['left_foot_strike', 'right_foot_strike']
        for col in strike_cols:
            values = df[col].values
            smoothed_strikes = self._smooth_categorical_signal(values, window=3)
            df[col] = smoothed_strikes
        
        self.frame_data = df.to_dict('records')
    
    def _smooth_signal(self, signal: np.ndarray, window: int = 5) -> np.ndarray:
        if len(signal) < window:
            return signal
        
        smoothed = np.copy(signal)
        half_window = window // 2
        
        for i in range(half_window, len(signal) - half_window):
            smoothed[i] = np.median(signal[i-half_window:i+half_window+1])
        
        return smoothed
    
    def _smooth_categorical_signal(self, signal: List[str], window: int = 3) -> List[str]:
        if len(signal) < window:
            return signal
        
        smoothed = list(signal)
        half_window = window // 2
        
        for i in range(half_window, len(signal) - half_window):
            window_values = signal[i-half_window:i+half_window+1]
            most_common = Counter(window_values).most_common(1)[0][0]
            smoothed[i] = most_common
        
        return smoothed
    
    def _save_analysis_results(self, video_name: str, output_dir: str):
        df = pd.DataFrame(self.frame_data)
        
        csv_path = os.path.join(output_dir, f"{video_name}_analysis.csv")
        df.to_csv(csv_path, index=False)
        
        json_path = os.path.join(output_dir, f"{video_name}_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.frame_data, f, indent=2, ensure_ascii=False)
    
    def _calculate_statistics(self, df: pd.DataFrame, video_name: str) -> Dict[str, Any]:
        return {
            'video_name': video_name,
            'total_frames': len(df),
            'duration_seconds': len(df) / self.video_fps,
            'analysis_timestamp': datetime.now().isoformat(),
            'foot_strike_stats': {
                'left_heel_percentage': (df['left_foot_strike'] == 'heel').mean() * 100,
                'left_midfoot_percentage': (df['left_foot_strike'] == 'midfoot').mean() * 100,
                'left_forefoot_percentage': (df['left_foot_strike'] == 'forefoot').mean() * 100,
                'right_heel_percentage': (df['right_foot_strike'] == 'heel').mean() * 100,
                'right_midfoot_percentage': (df['right_foot_strike'] == 'midfoot').mean() * 100,
                'right_forefoot_percentage': (df['right_foot_strike'] == 'forefoot').mean() * 100,
            },
            'posture_stats': {
                'avg_forward_lean': df['forward_lean'].mean(),
                'max_forward_lean': df['forward_lean'].max(),
                'min_forward_lean': df['forward_lean'].min(),
                'avg_head_forward': df['head_forward'].mean(),
                'max_head_forward': df['head_forward'].max(),
                'min_head_forward': df['head_forward'].min(),
                'lean_stability': df['forward_lean'].std()
            }
        }
    
    def _generate_report(self, video_name: str, output_dir: str):
        df = pd.DataFrame(self.frame_data)
        
        if df.empty:
            print("Brak danych do wygenerowania raportu")
            return
        
        stats = self._calculate_statistics(df, video_name)
        
        report_path = os.path.join(output_dir, f"{video_name}_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self._create_visualizations(df, video_name, output_dir)
        print(f"Raport zapisany: {report_path}")
    
    def _create_visualizations(self, df: pd.DataFrame, video_name: str, output_dir: str):
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        left_foot_counts = df['left_foot_strike'].value_counts()
        if not left_foot_counts.empty:
            axes[0, 0].pie(left_foot_counts.values, labels=left_foot_counts.index, 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Rozkład uderzeń - lewa stopa')
        
        right_foot_counts = df['right_foot_strike'].value_counts()
        if not right_foot_counts.empty:
            axes[0, 1].pie(right_foot_counts.values, labels=right_foot_counts.index, 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Rozkład uderzeń - prawa stopa')
        
        axes[1, 0].plot(df['timestamp'], df['forward_lean'], linewidth=1.5)
        axes[1, 0].set_title('Pochylenie do przodu w czasie')
        axes[1, 0].set_xlabel('Czas (sekundy)')
        axes[1, 0].set_ylabel('Kąt pochylenia (stopnie)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(df['timestamp'], df['head_forward'], linewidth=1.5, color='orange')
        axes[1, 1].set_title('Pozycja głowy w czasie')
        axes[1, 1].set_xlabel('Czas (sekundy)')
        axes[1, 1].set_ylabel('Kąt głowy (stopnie)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, f"{video_name}_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Wizualizacje zapisane: {viz_path}")

def validate_videos_directory(videos_dir: str) -> List[str]:
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
        print(f"Utworzono folder {videos_dir}. Umieść w nim pliki MP4.")
        return []
    
    video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith('.mp4')]
    
    if not video_files:
        print(f"Nie znaleziono plików MP4 w folderze {videos_dir}.")
        return []
    
    return video_files

def main():
    videos_dir = "videos"
    output_dir = "output2"
    
    video_files = validate_videos_directory(videos_dir)
    if not video_files:
        return
    
    print(f"Znaleziono {len(video_files)} plików MP4")
    
    with GaitPostureAnalyzer() as analyzer:
        for video_file in video_files:
            video_path = os.path.join(videos_dir, video_file)
            print(f"\n--- Przetwarzanie: {video_file} ---")
            
            try:
                output_video = analyzer.process_video(video_path, output_dir)
                print(f"Zakończono: {video_file}")
                
            except Exception as e:
                print(f"Błąd podczas przetwarzania {video_file}: {e}")
    
    print(f"\nAnaliza filmów zakończona!")
    print(f"Wyniki dostępne w folderze: {output_dir}")

if __name__ == "__main__":
    main()