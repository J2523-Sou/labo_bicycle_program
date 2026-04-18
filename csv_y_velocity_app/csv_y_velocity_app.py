import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import ttk as ttk2
import threading

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# mediapipe初期化
mp_pose = mp.solutions.pose

# 解析中止フラグ
stop_processing = False


# 処理停止フラグによる停止を実装（関数on_close）
def on_close():
    global stop_processing
    stop_processing = True
    root.destroy()

# メインウインドウの初期設定
root = tk.Tk()
root.title('【髙橋研究】解析ツール')
root.geometry('700x500')
root.protocol('WM_DELETE_WINDOW', on_close)

# 進捗バー
progressbar_frame = tk.Frame(root)
progressbar_frame.pack(fill='x', padx=20, pady=5)
overall_progress_var = tk.DoubleVar()
overall_progress_bar = ttk.Progressbar(progressbar_frame, variable=overall_progress_var, maximum=100)
overall_progress_bar.pack(fill='x', pady=(0,2))
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(progressbar_frame, variable=progress_var, maximum=100)
progress_bar.pack(fill='x')
overall_progress_label = tk.Label(progressbar_frame, text='全体進捗: 0/0')
overall_progress_label.pack(side='left', padx=10, anchor='n')

# ボタン
button_frame = tk.Frame(root)
button_frame.pack(pady=5)
select_button = tk.Button(button_frame, text='ファイル選択', command=lambda: select_files())
select_button.pack(side='left', padx=5)
param_button = tk.Button(button_frame, text='パラメータ設定', command=lambda: open_param_window())
param_button.pack(side='left', padx=5)
start_button = tk.Button(button_frame, text='スタート', command=lambda: start_analysis(), state='disabled')
start_button.pack(side='left', padx=5)
stop_button = tk.Button(button_frame, text='解析中止', command=lambda: stop_analysis(), state='disabled')
stop_button.pack(side='left', padx=5)

# --- 左右分割フレーム ---
top_frame = tk.Frame(root)
top_frame.pack(fill='x', padx=10, pady=2)

# チェックボックス
output_frame = tk.Frame(top_frame)
output_frame.pack(side='left', padx=2, pady=2, anchor='n')
graph_save_var = tk.BooleanVar(value=False)
excel_save_var = tk.BooleanVar(value=False)
video_save_var = tk.BooleanVar(value=False)
trajectory_draw_var = tk.BooleanVar(value=False)
tk.Checkbutton(output_frame, text='グラフ保存', variable=graph_save_var).pack(anchor='w')
tk.Checkbutton(output_frame, text='Excel保存', variable=excel_save_var).pack(anchor='w')
tk.Checkbutton(output_frame, text='動画保存', variable=video_save_var).pack(anchor='w')
tk.Checkbutton(output_frame, text='軌跡描画', variable=trajectory_draw_var).pack(anchor='w')

# 変動係数・真円度の表
all_cv_results = []
all_cv_titles = []
all_circleness = []  # 真円度履歴
cv_table_frame = tk.Frame(top_frame)
cv_table_frame.pack(side='left', padx=10, pady=2, anchor='n')
cv_table = ttk2.Treeview(cv_table_frame, columns=("filename", "cv", "circleness"), show="headings", height=6)
cv_table.heading("filename", text="動画ファイル")
cv_table.heading("cv", text="変動係数")
cv_table.heading("circleness", text="真円度")
cv_table.column("filename", width=180)
cv_table.column("cv", width=80)
cv_table.column("circleness", width=80)
cv_table.pack()

# データコピー関数（copy_cv_to_clipboard）
def copy_cv_to_clipboard():
    lines = ["動画ファイル\t変動係数\t真円度"]
    for title, cv, circ in zip(all_cv_titles, all_cv_results, all_circleness):
        lines.append(f"{title}\t{cv}\t{circ}")
    text = "\n".join(lines)
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()

# コピーボタン
copy_cv_btn = tk.Button(cv_table_frame, text="変動係数をコピー", command=copy_cv_to_clipboard)
copy_cv_btn.pack(pady=2)

output = tk.Frame(top_frame)
output.pack(side='right', expand=True, padx=10, pady=10)

# --- 進捗・標準偏差ラベルもボタンの下にまとめる ---
progress_label = tk.Label(output, text='ファイル選択待ち...')
progress_label.pack(pady=2)

# 標準偏差リアルタイム表示ラベル
std_label = tk.Label(output, text='標準偏差: -')
std_label.pack(pady=2)

# 真円度リアルタイム表示ラベル
circleness_label = tk.Label(output, text='真円度: -')
circleness_label.pack(pady=2)

# 経過時間リアルタイム表示ラベル
elapsed_time_label = tk.Label(output, text='経過時間: 0.0秒')
elapsed_time_label.pack(pady=2)


# ファイル選択とスタート制御
selected_files = []
def select_files():
    global stop_processing, selected_files
    stop_processing = False  # 新しいファイル選択時にリセット
    files = filedialog.askopenfilenames(
        title='動画ファイルを選択してください',
        filetypes=[('MP4 files', '*.mp4'), ('All files', '*.*')]
    )
    if not files:
        progress_label.config(text='ファイルが選択されませんでした')
        return
    selected_files = files
    progress_label.config(text=f'{len(files)}件のファイルが選択されました。スタートを押してください')
    stop_button.config(state='normal')  # 解析中止ボタンは有効化
    start_button.config(state='normal') # スタートボタン有効化

def start_analysis():
    global selected_files
    if not selected_files:
        progress_label.config(text='ファイルが選択されていません')
        return
    start_button.config(state='disabled')
    stop_button.config(state='normal')
    threading.Thread(target=process_files, args=(selected_files,), daemon=True).start()




# mediapipeパラメータ（精度選択系）を別ウインドウで設定
model_complexity_var = tk.IntVar(value=2)
min_detection_confidence_var = tk.DoubleVar(value=0.8)
min_tracking_confidence_var = tk.DoubleVar(value=0.8)

def open_param_window():
    param_win = tk.Toplevel(root)
    param_win.title('パラメータ設定')
    param_win.geometry('320x180')
    tk.Label(param_win, text='model_complexity').grid(row=0, column=0, sticky='e', padx=5, pady=5)
    model_complexity_menu = tk.OptionMenu(param_win, model_complexity_var, 0, 1, 2)
    model_complexity_menu.grid(row=0, column=1, padx=5, pady=5)
    tk.Label(param_win, text='min_detection_confidence').grid(row=1, column=0, sticky='e', padx=5, pady=5)
    min_detection_confidence_scale = tk.Scale(param_win, variable=min_detection_confidence_var, from_=0.0, to=1.0, resolution=0.05, orient='horizontal', length=150)
    min_detection_confidence_scale.grid(row=1, column=1, padx=5, pady=5)
    tk.Label(param_win, text='min_tracking_confidence').grid(row=2, column=0, sticky='e', padx=5, pady=5)
    min_tracking_confidence_scale = tk.Scale(param_win, variable=min_tracking_confidence_var, from_=0.0, to=1.0, resolution=0.05, orient='horizontal', length=150)
    min_tracking_confidence_scale.grid(row=2, column=1, padx=5, pady=5)
    tk.Button(param_win, text='閉じる', command=param_win.destroy).grid(row=3, column=0, columnspan=2, pady=10)




# 解析中止ボタン
def stop_analysis():
    global stop_processing
    stop_processing = True
    progress_label.config(text='解析を中止しています...')
    
# パラメータ設定ボタンをメインウインドウに追加



# --- ここは上部で定義済みなので削除 ---




# --- グラフと動画プレビュー横並び配置 ---
display_frame = tk.Frame(root)
display_frame.pack(fill='both', expand=True, padx=10, pady=10)

# グラフ領域
fig, ax1 = plt.subplots(figsize=(6,2.5))
ax2 = ax1.twinx()
line1, = ax1.plot([], [], color='tab:blue', label='velocity')
line2, = ax2.plot([], [], color='tab:orange', label='y')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Velocity', color='tab:blue')
ax2.set_ylabel('Y_location', color='tab:orange')
ax1.set_title('Velocity & Y_location')
lines = [line1, line2]
canvas = FigureCanvasTkAgg(fig, master=display_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side='left', fill='both', expand=True)

# 動画プレビュー領域
video_preview_label = tk.Label(display_frame)
video_preview_label.pack(side='left', padx=10, pady=5)
video_preview_img = None  # 参照保持用


def update_progress(filename, current, total, std, hensa_buffer, y_buffer, std_history, cv_history, window=100):
    # 経過時間は引数で渡す（デフォルトNone）
    elapsed_sec = None
    if hasattr(update_progress, 'start_time') and update_progress.start_time is not None:
        elapsed_sec = time.time() - update_progress.start_time
    # 各ラベルが存在する場合のみ更新
    if progress_label.winfo_exists():
        progress_label.config(text=f'{filename} {current}/{total} フレーム')
    if hasattr(progress_var, 'set'):
        progress_var.set(current / total * 100 if total > 0 else 0)
    if len(cv_history) > 0:
        cv_disp = f'{cv_history[-1]:.4f}'
    else:
        cv_disp = '-'
    if std_label.winfo_exists():
        std_label.config(text=f'標準偏差: {std:.4f}  変動係数: {cv_disp}')
    # --- 真円度リアルタイム計算・表示 ---
    try:
        global trajectory_points
        traj_arr = np.array(trajectory_points)
        if traj_arr.shape[0] > 10:
            center = np.mean(traj_arr, axis=0)
            radii = np.linalg.norm(traj_arr - center, axis=1)
            mean_r = np.mean(radii)
            std_r = np.std(radii)
            circleness = std_r / mean_r if mean_r != 0 else float('nan')
            circleness_disp = f'{circleness:.4f}' if not np.isnan(circleness) else '-'
        else:
            circleness_disp = '-'
    except Exception:
        circleness_disp = '-'
    if circleness_label.winfo_exists():
        circleness_label.config(text=f'真円度: {circleness_disp}')
    # 経過時間表示
    if elapsed_sec is not None and elapsed_time_label.winfo_exists():
        elapsed_time_label.config(text=f'経過時間: {elapsed_sec:.1f}秒')
    elif elapsed_time_label.winfo_exists():
        elapsed_time_label.config(text='経過時間: -')
    # グラフは常に描画
    x = range(max(0, len(hensa_buffer)-window), len(hensa_buffer))
    hensa_disp = hensa_buffer[-window:]
    y_disp = y_buffer[-window:]
    if len(hensa_disp) > 0:
        line1.set_data(x, hensa_disp)
        ax1.set_xlim(max(0, len(hensa_buffer)-window), len(hensa_buffer))
        ax1.set_ylim(0, max(hensa_disp)*1.1 if max(hensa_disp) > 0 else 1)
    if len(y_disp) > 0:
        line2.set_data(x, y_disp)
        ax2.set_ylim(min(y_disp)*0.9 if len(y_disp)>0 else 0, max(y_disp)*1.1 if len(y_disp)>0 else 1)
    ax2.set_xlim(ax1.get_xlim())
    # 凡例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    canvas.draw()
    root.update_idletasks()

def process_files(input_videos):
    global stop_processing
    total_files = len(input_videos)
    global video_preview_img
    global all_cv_results, all_cv_titles
    all_cv_results = []
    all_cv_titles = []
    cv_table.delete(*cv_table.get_children())
    for idx, input_video in enumerate(input_videos):
        # ファイルごとに処理開始時刻を記録
        update_progress.start_time = time.time()
        # 全体進捗バー更新
        overall_progress_var.set(idx / total_files * 100)
        # ファイルごとのcv履歴を記録
        file_cv = None
        file_title = input_video.split("/")[-1] if "/" in input_video else input_video.split("\\")[-1]
        if stop_processing:  # 中止フラグをチェック
            progress_label.config(text='解析が中止されました')
            stop_button.config(state='disabled')
            return
        
        # グラフ・進捗・履歴をリセット
        line1.set_data([], [])
        line2.set_data([], [])
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        canvas.draw()
        progress_var.set(0)
        std_label.config(text='標準偏差: -')
        root.update_idletasks()
        progress_label.config(text=f'Processing: {input_video}')
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            progress_label.config(text=f"Error: Could not open video: {input_video}")
            continue
        frame_count_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        # --- 動画出力用の初期化 ---
        if video_save_var.get():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video_path = input_video.rsplit('.', 1)[0] + '_output.mp4'
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        else:
            video_writer = None
        # 軌跡リストは毎回リセット
        global trajectory_points
        trajectory_points = []

        with mp_pose.Pose(
            static_image_mode = False,
            model_complexity = model_complexity_var.get(),
            min_detection_confidence = min_detection_confidence_var.get(),
            min_tracking_confidence = min_tracking_confidence_var.get()) as pose:

            target_landmark_id = mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
            velocity_buffer = deque(maxlen = 20)
            hensa_buffer = deque(maxlen = frame_count_video)
            y_buffer = deque(maxlen = frame_count_video)
            std_history = []  # 標準偏差履歴
            cv_history = []   # 変動係数履歴
            prev_time = time.time()
            prev_landmark_pos = None
            prev_velocity = None
            frame_count = 0
            total_velocity = 0


            # 以下処理系統
            while True:
                if stop_processing:  # フレーム処理中も中止フラグをチェック
                    cap.release()
                    if video_writer:
                        video_writer.release()
                    progress_label.config(text='解析が中止されました')
                    stop_button.config(state='disabled')
                    return
                
                success, inage = cap.read()
                if not success:
                    progress_label.config(text="Processing Completed")
                    break
                
                frame_count += 1
                image_rgb = cv2.cvtColor(inage, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                current_time = time.time()
                # delta_time = current_time - prev_time
                prev_time = current_time
                current_landmark_pos = None
                if results.pose_landmarks:
                    landmark = results.pose_landmarks.landmark[target_landmark_id]
                    current_x_px = int(landmark.x * inage.shape[1])
                    current_y_px = int(landmark.y * inage.shape[0])
                    current_landmark_pos = np.array([current_x_px, current_y_px])
                    # 追跡点の軌跡
                    trajectory_points.append((current_x_px, current_y_px))
                
                # y座標外れ値判定用: 直近のy値リストを作成                
                if current_landmark_pos is not None and prev_landmark_pos is not None:
                    y_arr = np.array(y_buffer) if len(y_buffer) > 20 else None
                    y_val = landmark.y
                    is_outlier = False
                    if y_arr is not None and len(y_arr) > 0:
                        y_median = np.median(y_arr)
                        y_std = np.std(y_arr)
                        if y_std > 0 and np.abs(y_val - y_median) > 3 * y_std:
                            is_outlier = True
                    # 外れ値でなければバッファに追加
                    if not is_outlier:
                        diff = current_landmark_pos - prev_landmark_pos
                        diff_norm = np.linalg.norm(diff)
                        velocity_buffer.append(diff_norm)
                        smoothed_value = np.mean(velocity_buffer)

                        hensa_buffer.append(smoothed_value)
                            
                        y_buffer.append(y_val)
                        
                prev_landmark_pos = current_landmark_pos

                if hensa_buffer is not None:
                    std_now = np.std(list(hensa_buffer))
                    mean_now = np.mean(list(hensa_buffer))
                    cv_now = std_now / mean_now if mean_now != 0 else float('nan')
                    std_history.append(std_now)
                    cv_history.append(cv_now)
                    file_cv = cv_now
                
                
                root.after(0, update_progress, input_video, frame_count, frame_count_video, std_now, list(hensa_buffer), list(y_buffer), list(std_history), list(cv_history))

                # --- 動画プレビュー ---
                # ランドマーク点・軌跡を描画
                preview_img = inage.copy()
                if results.pose_landmarks:
                    cx, cy = current_x_px, current_y_px
                    cv2.circle(preview_img, (cx, cy), 15, (0, 0, 255), -1)
                # 軌跡描画ON時のみ描画
                if trajectory_draw_var.get() and trajectory_points and len(trajectory_points) > 1:
                    for i in range(1, len(trajectory_points)):
                        cv2.line(preview_img, trajectory_points[i-1], trajectory_points[i], (0,255,0), 5)
                # Tkinter用に変換（root.afterで安全に画像生成・表示）
                def set_preview_image():
                    global video_preview_img
                    if video_preview_label.winfo_exists():
                        preview_img_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(preview_img_rgb)
                        pil_img = pil_img.resize((480, 360))
                        video_preview_img = ImageTk.PhotoImage(pil_img)
                        video_preview_label.config(image=video_preview_img)
                        video_preview_label.image = video_preview_img  # 参照保持
                root.after(0, set_preview_image)

                # --- 動画出力 ---
                if video_writer is not None:
                    video_writer.write(preview_img)

            # ループ終了時に必ず最終GUI更新
            root.after(0, update_progress, input_video, frame_count, frame_count_video, std_now, list(hensa_buffer), list(y_buffer), list(std_history), list(cv_history))
            # ファイルごとの最終cvを記録
            if file_cv is not None and not np.isnan(file_cv):
                all_cv_results.append(file_cv)
            else:
                all_cv_results.append('-')
            all_cv_titles.append(file_title)
            # --- 真円度評価 ---
            if len(trajectory_points) > 10:
                traj_arr = np.array(trajectory_points)
                center = np.mean(traj_arr, axis=0)
                radii = np.linalg.norm(traj_arr - center, axis=1)
                mean_r = np.mean(radii)
                std_r = np.std(radii)
                circleness = std_r / mean_r if mean_r != 0 else float('nan')
                circleness_disp = f"{circleness:.4f}" if not np.isnan(circleness) else '-'
            else:
                circleness_disp = '-'
            all_circleness.append(circleness_disp)
            # Treeview表に反映
            cv_table.insert("", "end", values=(file_title, file_cv if file_cv is not None and not np.isnan(file_cv) else '-', circleness_disp))
            # 全体進捗ラベルは件数のみ
            overall_progress_label.config(text=f'全体進捗: {idx+1}/{total_files}')

        if video_writer:
            video_writer.release()

        # Excel保存ON時のみ保存（動画保存の有無に関係なく）
        if excel_save_var.get():
            min_len = min(len(hensa_buffer), len(y_buffer), len(std_history), len(cv_history))
            df = pd.DataFrame({
                'frame': list(range(min_len)),
                'velocity': list(hensa_buffer)[:min_len],
                'y': list(y_buffer)[:min_len],
                'std': list(std_history)[:min_len],
                'cv': list(cv_history)[:min_len]
            })
            # 真円度もExcelに追記
            df2 = pd.DataFrame({'filename': [file_title], 'circleness': [circleness_disp]})
            out_name = input_video.rsplit('.', 1)[0] + '_output.xlsx'
            with pd.ExcelWriter(out_name) as writer:
                df.to_excel(writer, index=False, sheet_name='data')
                df2.to_excel(writer, index=False, sheet_name='circleness')
        # グラフ保存ON時のみ保存（動画保存の有無に関係なく）
        if graph_save_var.get():
            fig.savefig(input_video.rsplit('.', 1)[0] + '_output.png')
        cap.release()
    
    # 全ファイル処理完了時
    if not stop_processing:
        progress_label.config(text='全ファイル処理完了')
    std_label.config(text='標準偏差: -')
    stop_button.config(state='disabled')  # 処理完了時にボタンを無効化



root.mainloop()