import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import os
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import msvcrt
import matplotlib.pyplot as plt

# mediapipe初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


import os
# ファイル選択ダイアログ（複数選択）
root = tk.Tk()
root.withdraw()
input_videos = filedialog.askopenfilenames(
    title='動画ファイルを選択してください（複数可）',
    filetypes=[('MP4 files', '*.mp4'), ('All files', '*.*')]
)
if not input_videos:
    print('ファイルが選択されませんでした。処理を終了します。')
    exit()

    # ...existing code...

# 複数ファイルをforループで処理
for input_video in input_videos:
    print(f'処理開始: {input_video}')
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_video}")
        continue
    frame_count_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    with mp_pose.Pose(
        static_image_mode = False,
        model_complexity = 2,
        min_detection_confidence = 0.9,
        min_tracking_confidence = 0.9) as pose:

        target_landmark_id = mp_pose.PoseLandmark.LEFT_HEEL.value
        velocity_buffer = deque(maxlen = 10)
        hensa_buffer = deque(maxlen = frame_count_video)
        velocity_magnitude_list = []
        landmark_y_list = []
        prev_time = time.time()
        prev_landmark_pos = None
        prev_velocity = None
        frame_count = 0
        total_velocity = 0

        pbar = tqdm(total=frame_count_video, desc=f"Processing frames ({os.path.basename(input_video)})")
        while True:
            success, image = cap.read()
            if not success:
                print("Processing Completed")
                break

            frame_count += 1
            pbar.update(1)

            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'q':
                    print("qキーが押されたため中止します")
                    break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True
            current_time = time.time()
            delta_time = current_time - prev_time
            prev_time = current_time
            current_landmark_pos = None
            current_velocity = None
            landmark_x = None
            landmark_y = None
            landmark_x_before = None
            landmark_y_before = None

            if results.pose_landmarks:
                landmark = results.pose_landmarks.landmark[target_landmark_id]
                landmark_x = landmark.x
                landmark_y = landmark.y
                current_x_px = int(landmark.x * image.shape[1])
                current_y_px = int(landmark.y * image.shape[0])
                current_landmark_pos = np.array([current_x_px, current_y_px])
                current_x_px_before = current_x_px
                current_y_px_before = current_y_px

            if current_landmark_pos is not None and prev_landmark_pos is not None:
                landmark = results.pose_landmarks.landmark[target_landmark_id]
                landmark_x = landmark.x
                landmark_y = landmark.y
                current_velocity = (current_landmark_pos - prev_landmark_pos) / delta_time
                velocity_buffer.append(current_velocity)
                smoothed_velocity = np.mean(velocity_buffer, axis=0)
                velocity_magnitude = np.linalg.norm(smoothed_velocity)
                velocity_magnitude_list.append(velocity_magnitude)
                landmark_y_list.append(landmark_y)
                hensa_buffer.append(velocity_magnitude)

            prev_landmark_pos = current_landmark_pos
            landmark_x_before = landmark_x
            landmark_y_before = landmark_y
            if frame_count == 1:
                print("Processing Started")

        pbar.close()

        # グラフ画像名を動画ファイルごとに変更
        graph_filename = os.path.splitext(os.path.basename(input_video))[0] + '_graph.png'
        fig, ax1 = plt.subplots(figsize=(16, 6))
        x = range(len(velocity_magnitude_list))
        color1 = 'tab:blue'
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Velocity Magnitude', color=color1)
        ax1.plot(x, velocity_magnitude_list, color=color1, label='Velocity')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Landmark Y', color=color2)
        ax2.plot(x, landmark_y_list, color=color2, label='Landmark Y')
        ax2.tick_params(axis='y', labelcolor=color2)
        fig.tight_layout()
        plt.title('Velocity Magnitude & Landmark Y')
        plt.savefig(graph_filename)
        plt.close()
        print(f'グラフ画像を {graph_filename} として保存しました')
        results_hensa = np.std(list(hensa_buffer)) / np.mean(list(hensa_buffer))
        print(results_hensa)
        cap.release()
    print(results_hensa)
    cap.release()
        
        