import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
import glob
import os

# Tkinter権限取得
root = tk.Tk()
root.withdraw()

# 現在の解析動画番号
count_videos = 0

# ファイル選択GUIをTkinterで表示
video_files = filedialog.askopenfilenames(
    title="解析する動画を選択",
    filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")]
)

# ファイルが選択されなかった場合は終了
if not video_files:
    print("No files selected. 終了します。")
    exit(0)

# ファイルパスを固定したい場合はこちら（ディレクトリ内videos）
# # 処理する動画一覧
# video_files = glob.glob("videos/*.mp4")

# MediaPipe Pose 初期化
mp_pose = mp.solutions.pose

# パラメータ
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# 保存フォルダ
os.makedirs("trajectory", exist_ok=True)

# 選択ファイル数回繰り返す
for video_path in video_files:
    
    # 動画番号をインクリメントする
    count_videos += 1

    # 動画ファイルのサイズリセット
    width = 0
    height = 0

    # 動画の幅と高さを取得(絶対座標)
    width = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"Processing: {video_path}")
    
    # 動画を開く
    cap = cv2.VideoCapture(video_path)
    
    # 画像保存用のベース名
    base = os.path.splitext(os.path.basename(video_path))[0]


    # 前フレームの座標
    prev_point = None
    frame_idx = 0
    last_frame = None

    # フレーム数取得
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    while True:
        
        # フレームを開いているか/その内容を取得
        ret, frame = cap.read()
        
        # もしフレームを取得できていなければ終了する
        if not ret:
            break

        # RGB に変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Pose で処理
        results = pose.process(frame_rgb)

        # --- 左つま先（LEFT_FOOT_INDEX） ---
        landmark = None
        pixel_point = None
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            landmark = (lm.x, lm.y)
            # 画素座標へ変換
            pixel_point = (int(lm.x * width), int(lm.y * height))

        # 線を描画（前フレームと現在フレーム両方が有効な場合のみ）
        if prev_point is not None and pixel_point is not None:
            cv2.line(frame, prev_point, pixel_point, (0, 255, 0), 2)

        # 進捗表示用の座標文字列（CSVは出力しない）
        x_value = f"{landmark[0] * width:.6f}" if landmark else ""
        y_value = f"{landmark[1] * height:.6f}" if landmark else ""


        # 進行度インジケータ
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

        # 次フレーム用に現在の画素座標を保持
        prev_point = pixel_point
        frame_idx += 1
        # 最終フレーム保存用に更新
        last_frame = frame.copy()

    cap.release()

    # 最終フレームを画像として保存
    if last_frame is not None:
        last_img_path = f"results/{base}_lastframe.png"
        cv2.imwrite(last_img_path, last_frame)
        print(f"  → saved last frame image: {last_img_path}")
