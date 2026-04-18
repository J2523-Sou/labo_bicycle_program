import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import os

# mediapipe初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 動画パス指定
output_video = 'kaiseki_after.mp4'
input_video = 'kaiseki.mp4'

print(os.path.exists(input_video))  # TrueならOK、Falseならパスが間違い


# 読み込み
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    
# 動画情報取得
frame_count_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 動画出力設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

with mp_pose.Pose(
    static_image_mode = False,
    model_complexity = 2,
    min_detection_confidence = 0.9,
    min_tracking_confidence = 0.9) as pose:

    target_landmark_id = mp_pose.PoseLandmark.LEFT_HEEL.value
    
    # バッファ
    velocity_buffer = deque(maxlen = 10)
    hensa_buffer = deque(maxlen = frame_count_video)
    
    # 経過時間
    prev_time = time.time()
    
    # 位置・速度
    prev_landmark_pos = None
    prev_velocity = None
    
    
    
    
    # 処理中のフレーム
    frame_count = 0
    
    total_velocity = 0
    
    while True:
        success, image = cap.read()
        if not success:
            print("Processing Completed")
            break
        
        frame_count += 1
        
        # 画像形式変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 書き込み不可
        image_rgb.flags.writeable = False
        
        # 骨格推定結果をresultsへ格納
        results = pose.process(image_rgb)
        
        # 書き込み可
        image_rgb.flags.writeable = True
        
        # 時間操作
        current_time = time.time()              # 現在時間を取得
        delta_time = current_time - prev_time   # 時間差を算出
        prev_time = current_time                # 前回時間を更新
        
        current_landmark_pos = None
        current_velocity = None
        landmark_x = None
        landmark_y = None
        landmark_x_before = None
        landmark_y_before = None

        # 骨格推定成功時
        if results.pose_landmarks:
            landmark = results.pose_landmarks.landmark[target_landmark_id]
            landmark_x = landmark.x
            landmark_y = landmark.y
            current_x_px = int(landmark.x * image.shape[1])         # 各座標をpxに変換
            current_y_px = int(landmark.y * image.shape[0])

            
            # 位置をNumpy配列に格納
            current_landmark_pos = np.array([current_x_px, current_y_px])
            current_x_px_before = current_x_px
            current_y_px_before = current_y_px
            
            # ランドマークを描画
            cv2.circle(image, (current_x_px, current_y_px), 8, (0, 255, 0), -1)
            
            # それ以外を描画
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            
        
        # 現在と一つ前の処理が成功している場合
        if current_landmark_pos is not None and prev_landmark_pos is not None:
            
            landmark = results.pose_landmarks.landmark[target_landmark_id]
            landmark_x = landmark.x
            landmark_y = landmark.y
            
            # 速度算出
            current_velocity = (current_landmark_pos - prev_landmark_pos) / delta_time
            velocity_buffer.append(current_velocity)                    # current_velocityをvelocity_bufferに追加
            smoothed_velocity = np.mean(velocity_buffer, axis=0)        # 平均化
            
            velocity_magnitude = np.linalg.norm(smoothed_velocity)      # smoothed_velocityよりベクトルを算出
            
           
            
            hensa_buffer.append(velocity_magnitude)                     # velocity_magnitudeをhensa_bufferに追加
            
        # ランドマーク位置を更新
        prev_landmark_pos = current_landmark_pos
        
        landmark_x_before = landmark_x
        landmark_y_before = landmark_y
        
        
        if frame_count == 1:
            print("Processing Started")
            
        print(frame_count, end='/' ,flush=True)
        print(frame_count_video,flush=True)

        # if frame_count == frame_count_video * 0.1:
        #     print("==------------------10%")
            
        # if frame_count == frame_count_video * 0.2:
        #     print("====----------------20%")
            
        # if frame_count == frame_count_video * 0.3:
        #     print("======--------------30%")
            
        # if frame_count == frame_count_video * 0.4:
        #     print("========------------40%")
            
        # if frame_count == frame_count_video * 0.5:
        #     print("==========----------50%")
            
        # if frame_count == frame_count_video * 0.6:
        #     print("============--------60%")
    
        # if frame_count == frame_count_video * 0.7:
        #     print("==============------70%")
    
        # if frame_count == frame_count_video * 0.8:
        #     print("================----80%")
            
        # if frame_count == frame_count_video * 0.9:
        #     print("==================--90%")
            
        # if frame_count == frame_count_video * 1.0:
        #     print("====================100%")

        #average_velocity = total_velocity / frame_count
        
        out.write(image)  # 最後のフレームを書き込み
        
results_hensa = np.std(list(hensa_buffer)) / np.mean(list(hensa_buffer))
        
print(results_hensa)
        
cap.release()
out.release()
        
        