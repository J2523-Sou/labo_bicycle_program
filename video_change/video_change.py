import cv2
import tqdm

# 元の動画を読み込み
cap = cv2.VideoCapture("input/input.MOV")

# 新しい設定
new_width = 1920
new_height = 1080
new_fps = 240.0  # ここでfpsを変更
now_frame = 0

# 保存用の設定（VideoWriter）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/output.mp4', fourcc, new_fps, (new_width, new_height))

with tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        now_frame += 1
        pbar.update(1)

        # 解像度の変更（リサイズ）
        resized_frame = cv2.resize(frame, (new_width, new_height))

        out.write(resized_frame)
        

cap.release()
out.release()