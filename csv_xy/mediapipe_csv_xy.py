import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
import glob
import csv
import os
import math

root = tk.Tk()
root.withdraw()

count_videos = 0

video_files = filedialog.askopenfilenames(
    title="解析する動画を選択",
    filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")]
)

if not video_files:
    print("No files selected. 終了します。")
    exit(0)

# ファイルパスを固定したい場合はこちら（ディレクトリ内videos）
# # 処理する動画一覧
# video_files = glob.glob("videos/*.mp4")

# MediaPipe Pose 初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,   # 1.0 は高すぎるので緩和
    min_tracking_confidence=0.5     # 追跡閾値も緩和
)


# 保存フォルダ
os.makedirs("results", exist_ok=True)


for video_path in video_files:
    
    count_videos += 1

    width = 0
    height = 0

    width = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # 出力 CSV 名
    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = f"results/{base}.csv"

    # 前フレームの座標
    prev_point = None
    frame_idx = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y"])

        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # RGB に変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # --- 左つま先（LEFT_FOOT_INDEX） ---
            landmark = None
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                landmark = (lm.x, lm.y)
                if frame_idx % 30 == 0:
                    print(f"landmark frame={frame_idx} x={lm.x:.3f} y={lm.y:.3f}")
            else:
                if frame_idx % 30 == 0:
                    print(f"no landmarks at frame={frame_idx}")

            

            # 座標を出力（検出なしは空セルにする）
            x_value = f"{landmark[0] * width:.6f}" if landmark else ""
            y_value = f"{landmark[1] * height:.6f}" if landmark else ""
            writer.writerow([frame_idx, x_value, y_value])

            print("now : ", end="")
            print(frame_idx, end="")
            print(" / ", end="")
            print(total_frame, end="")
            print(" / ", end="")
            print(x_value, end="")
            print(" / ", end="")
            print(y_value, end="")
            print("/", end=" ")
            print(count_videos, end="")
            print("/", end=" ")
            print(len(video_files))

            prev_point = landmark
            frame_idx += 1

    cap.release()
    print(f"  → saved CSV: {csv_path}")
