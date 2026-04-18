import mediapipe as mp
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from openpyxl import Workbook
from collections import deque
from tqdm import tqdm
import os

def save_toe_coordinates_to_excel(toe_data, video_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "ToeCoordinates"
    ws.append(["Frame", "LeftToe_X_px", "LeftToe_Y_px", "LeftDist", "SpeedBufferAvg"])
    for row in toe_data:
        ws.append(row)
    # 動画ファイル名からExcel名を生成
    base = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.dirname(video_path)
    save_path = os.path.join(save_dir, f"{base}_analysis.xlsx")
    wb.save(save_path)
    print(f"つま先座標を {save_path} に保存しました")

def select_files():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    return list(file_paths)

def process_video(file_path, min_detection_confidence=0.9, min_tracking_confidence=0.9):
    cap = cv2.VideoCapture(file_path)
    
    # 動画情報の取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
    mp_draw = mp.solutions.drawing_utils

    toe_data = []
    prev_left = None
    speed_buffer = deque(maxlen=10)

    with tqdm(total=total_frames, desc="進捗", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            if results.pose_landmarks:
                
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                left_toe = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                
                # ユークリッド距離計算
                if prev_left is not None:
                    left_dist = ((left_toe.x - prev_left[0]) ** 2 + (left_toe.y - prev_left[1]) ** 2) ** 0.5
                    speed_buffer.append(left_dist)
                else:
                    left_dist = 0.0
                    speed_buffer.append(left_dist)
                    
                speed_buffer_avg = np.mean(speed_buffer)
                
                toe_data.append([
                    cap.get(cv2.CAP_PROP_POS_FRAMES),  # フレーム番号
                    left_toe.x * width, left_toe.y * height,
                    left_dist,
                    speed_buffer_avg
                ])
                prev_left = (left_toe.x, left_toe.y)

            pbar.update(1)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

    save_toe_coordinates_to_excel(toe_data, file_path)

def main():
    file_paths = select_files()
    if file_paths:
        for file_path in file_paths:
            process_video(file_path)
    else:
        print("ファイルが選択されませんでした")

# ファイルが直接実行された場合、関数mainを呼び出す
if __name__ == "__main__":
    main()
