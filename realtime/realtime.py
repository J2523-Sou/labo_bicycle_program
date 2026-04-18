import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque # 過去のデータを効率よく管理するためのツール

# --- MediaPipeの設定 ---
mp_pose = mp.solutions.pose # 全身の姿勢推定用
mp_drawing = mp.solutions.drawing_utils # 検出結果を描画するツール
mp_drawing_styles = mp.solutions.drawing_styles # 描画スタイル

# カメラの準備
cap = cv2.VideoCapture(0) # 0はPCに接続されたデフォルトのカメラ

# MediaPipe Poseモデルの初期化
# model_complexity=1: モデルの複雑さ。1はバランスの取れた設定。
# min_detection_confidence=0.5: 検出の確信度が50%以上の場合に検出とみなす。
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9) as pose:

    # --- 加速度計算と一定性評価のための変数 ---
    # どの関節の加速度を測るか指定（例: 右手首）
    # 他の関節を試したい場合は、mp_pose.PoseLandmark.LEFT_ANKLE.value などに変更
    target_landmark_id = mp_pose.PoseLandmark.RIGHT_HEEL.value 

    # 速度の計算時に使う、過去の速度データの保存場所（平均を取ってノイズを減らすため）
    velocity_history = deque(maxlen=30) # 直近5フレーム分の速度を保持

    # 加速度の一定性を測るために、過去の加速度データを保存する場所
    # ここでは直近60フレーム（約2秒分、30fpsの場合）の加速度を保存
    acceleration_x_buffer = deque(maxlen=30) 
    acceleration_y_buffer = deque(maxlen=30) 
    
    velocity_buffer = deque(maxlen=60)

    # 前のフレームの時刻と位置、速度を記録しておく変数
    prev_time = time.time() 
    prev_landmark_pos = None # 前のフレームの右手首のピクセル座標 (X, Y)
    prev_velocity = None     # 前のフレームの右手首の速度 (Vx, Vy)

    # --- メインループ：カメラからの映像を1フレームずつ処理 ---
    while cap.isOpened():
        success, image = cap.read() # 1フレーム読み込み
        if not success:
            print("カメラのフレームが空です。")
            continue

        # 画像の準備（MediaPipeが処理しやすい形式に変換）
        # 左右反転し、BGRからRGBに変換
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # 処理を速くするために一時的に書き込み不可に
        
        # 姿勢推定の実行
        results = pose.process(image)

        # 画像を元の形式に戻す
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_time = time.time() # 現在の時刻
        delta_time = current_time - prev_time # 前のフレームからの時間差（秒）
        prev_time = current_time # 次のフレームのために時刻を更新

        current_landmark_pos = None # 現在の右手首のピクセル座標

        if results.pose_landmarks: # 姿勢が検出された場合
            # 指定した関節の座標を取得
            lm = results.pose_landmarks.landmark[target_landmark_id]
            
            # 正規化された座標（0〜1）を実際のピクセル座標に変換
            current_x_px = int(lm.x * image.shape[1])
            current_y_px = int(lm.y * image.shape[0])
            current_landmark_pos = np.array([current_x_px, current_y_px]) # NumPy配列に変換

            # 検出した右手首を黄色い丸で描画（デバッグ用）
            cv2.circle(image, (current_x_px, current_y_px), 8, (255, 0, 0), -1) 
            
            # 全身の骨格を描画
            # mp_drawing.draw_landmarks(
            #     image,
            #     results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS, # 関節をつなぐ線
            #     mp_drawing_styles.get_default_pose_landmarks_style()) # 描画スタイル

        # --- 速度と加速度の計算 ---
        current_velocity = None
        current_acceleration = None

        # 過去のランドマーク位置と時間差がある場合のみ計算
        if current_landmark_pos is not None and prev_landmark_pos is not None and delta_time > 0:
            # 1. 速度の計算： (今の位置 - 前の位置) / 時間差
            current_velocity = (current_landmark_pos - prev_landmark_pos) / delta_time
            velocity_history.append(current_velocity) # 速度履歴に追加

            # 速度の移動平均を計算（ノイズを減らすため）
            smoothed_velocity = np.mean(list(velocity_history), axis=0)
            smoothed_velocity_text_x = f"{smoothed_velocity[0]}"
            smoothed_velocity_text_y = f"{smoothed_velocity[1]}"
            cv2.putText(image, smoothed_velocity_text_x, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # 赤文字で表示
            cv2.putText(image, smoothed_velocity_text_y, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # 赤文字で表示
            velocity = abs(smoothed_velocity[0]) + abs(smoothed_velocity[1])
            velocity_text = f"{velocity}"
            cv2.putText(image, velocity_text, (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # 赤文字で表示
            
            velocity_buffer.append(velocity)
            
            velocity_hensa = np.std(list(velocity_buffer))
            velocity_hensa_text = f"{velocity_hensa}"
            cv2.putText(image, velocity_hensa_text, (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # 赤文字で表示


            # # 前の速度データがある場合のみ加速度を計算
            # if prev_velocity is not None:
            #     # 2. 加速度の計算： (今の速度 - 前の速度) / 時間差
            #     current_acceleration = (smoothed_velocity - prev_velocity) / delta_time
                
            #     # 計算された加速度を履歴バッファに保存
            #     acceleration_x_buffer.append(current_acceleration[0])
            #     acceleration_y_buffer.append(current_acceleration[1])

            #     # 3. 加速度を画面に表示
            #     accel_text = f"Acc_X: {current_acceleration[0]:.2f} px/s^2, Acc_Y: {current_acceleration[1]:.2f} px/s^2"
                

            #     # 4. 加速度の「一定性」を評価（標準偏差を計算）
            #     # 加速度の履歴が十分に溜まったら計算開始
            #     if len(acceleration_x_buffer) == acceleration_x_buffer.maxlen:
            #         std_dev_ax = np.std(list(acceleration_x_buffer)) # X方向加速度の標準偏差
            #         std_dev_ay = np.std(list(acceleration_y_buffer)) # Y方向加速度の標準偏差

            #         # 標準偏差を画面に表示
            #         std_dev_text = f"StdDev_Ax: {std_dev_ax:.2f}, StdDev_Ay: {std_dev_ay:.2f}"
            #         cv2.putText(image, std_dev_text, (10, 60), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # 青文字で表示
                    
            #         # 標準偏差が小さいほど「一定」と判断
            #         # ここで設定されている50はあくまで仮の「しきい値」です。
            #         # 実際に動かしてみて、適切な値を見つけてください。
            #         if std_dev_ax < 100 and std_dev_ay < 100: 
            #         # if std_dev_ax < 500:
            #              cv2.putText(image, "Acceleration is Relatively Constant!", (10, 90), 
            #                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # 緑文字
            #         else:
            #              cv2.putText(image, "Acceleration is NOT Constant.", (10, 90), 
            #                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # オレンジ文字
                         
            #         # if cv2.waitKey(1) & 0xFF == ord('q'):
                        

            prev_velocity = smoothed_velocity # 今回の平滑化された速度を次のフレームのために保存
        
        prev_landmark_pos = current_landmark_pos # 今回のランドマーク位置を次のフレームのために保存

        # 処理結果を表示
        cv2.imshow('MediaPipe Pose with Acceleration & Constancy', image)

        # 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 終了処理
cap.release() # カメラを解放
cv2.destroyAllWindows() # ウィンドウを閉じる