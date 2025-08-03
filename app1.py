import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, List

class GaitPostureAnalyzer:
    
    FOOT_STRIKE_THRESHOLD = 5
    MIDFOOT_THRESHOLD = 10
    
    def __init__(self, detection_confidence: float = 0.5, tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.frame_data: List[Dict[str, Any]] = []
        self.current_frame: int = 0
        self.video_fps: float = 30.0
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pose.close()
    
    def _classify_foot_strike(self, heel_diff: float, toe_diff: float) -> str:
        if abs(heel_diff) < self.FOOT_STRIKE_THRESHOLD:
            return "heel"
        elif abs(toe_diff) < self.FOOT_STRIKE_THRESHOLD:
            return "forefoot"
        elif abs(heel_diff - toe_diff) < self.MIDFOOT_THRESHOLD:
            return "midfoot"
        else:
            return "heel" if heel_diff < toe_diff else "forefoot"
    
    def analyze_foot_strike(self, landmarks, frame_width: int, frame_height: int) -> Tuple[str, str]:
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
            
            left_heel_diff = left_heel_y - left_ankle_y
            left_toe_diff = left_foot_y - left_ankle_y
            right_heel_diff = right_heel_y - right_ankle_y
            right_toe_diff = right_foot_y - right_ankle_y
            
            left_strike_type = self._classify_foot_strike(left_heel_diff, left_toe_diff)
            right_strike_type = self._classify_foot_strike(right_heel_diff, right_toe_diff)
            
            return left_strike_type, right_strike_type
            
        except (IndexError, AttributeError):
            return "unknown", "unknown"
    
    def analyze_posture_lean(self, landmarks) -> Tuple[float, float]:
        try:
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            
            torso_angle = np.arctan2(shoulder_center_x - hip_center_x, hip_center_y - shoulder_center_y)
            torso_angle_degrees = np.degrees(torso_angle)
            
            head_shoulder_angle = np.arctan2(nose.x - shoulder_center_x, shoulder_center_y - nose.y)
            head_angle_degrees = np.degrees(head_shoulder_angle)
            
            forward_lean = abs(torso_angle_degrees)
            head_forward = abs(head_angle_degrees)
            
            return forward_lean, head_forward
            
        except (IndexError, AttributeError):
            return 0.0, 0.0
    
    def _create_frame_info(self, frame_number: int) -> Dict[str, Any]:
        return {
            'frame_number': frame_number,
            'timestamp': frame_number / self.video_fps,
            'left_foot_strike': 'unknown',
            'right_foot_strike': 'unknown',
            'forward_lean': 0.0,
            'head_forward': 0.0
        }
    
    def _draw_annotations(self, frame, results, left_strike: str, right_strike: str, 
                         forward_lean: float, head_forward: float):
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
            
            self.frame_data = []
            self.current_frame = 0
            
            print(f"Rozpoczynam analizę filmu: {video_name}")
            print(f"Całkowita liczba klatek: {total_frames}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                frame_info = self._create_frame_info(self.current_frame)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    left_strike, right_strike = self.analyze_foot_strike(landmarks, frame_width, frame_height)
                    forward_lean, head_forward = self.analyze_posture_lean(landmarks)
                    
                    frame_info.update({
                        'left_foot_strike': left_strike,
                        'right_foot_strike': right_strike,
                        'forward_lean': forward_lean,
                        'head_forward': head_forward
                    })
                    
                    self._draw_annotations(frame, results, left_strike, right_strike, 
                                         forward_lean, head_forward)
                
                self.frame_data.append(frame_info)
                out.write(frame)
                self.current_frame += 1
                
                if self.current_frame % 30 == 0:
                    progress = (self.current_frame / total_frames) * 100
                    print(f"Przetworzono {self.current_frame}/{total_frames} klatek ({progress:.1f}%)")
            
            print("Analiza zakończona.")
            
        finally:
            cap.release()
            out.release()
        
        self._save_analysis_results(video_name, output_dir)
        self._generate_report(video_name, output_dir)
        
        print(f"Wyniki zapisane w folderze: {output_dir}")
        return output_video_path
    
    def _save_analysis_results(self, video_name: str, output_dir: str):
        df = pd.DataFrame(self.frame_data)
        
        csv_path = os.path.join(output_dir, f"{video_name}_analysis.csv")
        df.to_csv(csv_path, index=False)
        
        json_path = os.path.join(output_dir, f"{video_name}_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.frame_data, f, indent=2, ensure_ascii=False)
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            'video_name': '',
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
                'avg_head_forward': df['head_forward'].mean(),
                'max_head_forward': df['head_forward'].max(),
                'lean_stability': df['forward_lean'].std()
            }
        }
    
    def _generate_report(self, video_name: str, output_dir: str):
        df = pd.DataFrame(self.frame_data)
        
        if df.empty:
            print("Brak danych do wygenerowania raportu")
            return
        
        stats = self._calculate_statistics(df)
        stats['video_name'] = video_name
        
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
    output_dir = "output1"
    
    video_files = validate_videos_directory(videos_dir)
    if not video_files:
        return
    
    with GaitPostureAnalyzer() as analyzer:
        for video_file in video_files:
            video_path = os.path.join(videos_dir, video_file)
            print(f"\n--- Przetwarzanie: {video_file} ---")
            
            try:
                output_video = analyzer.process_video(video_path, output_dir)
                print(f"Zakończono: {video_file}")
                
            except Exception as e:
                print(f"Błąd podczas przetwarzania {video_file}: {e}")
    
    print(f"\nAnaliza  filmów zakończona")
    print(f"Wyniki dostępne w folderze: {output_dir}")

if __name__ == "__main__":
    main()