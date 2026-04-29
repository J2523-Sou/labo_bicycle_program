import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class VideoDualAxisApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Foot Y / Speed Analyzer")
        self.root.geometry("1280x760")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_path = ""
        self.processing_thread = None
        self.message_queue: queue.Queue[dict] = queue.Queue()
        self.stop_event = threading.Event()

        self.frame_history: deque[int] = deque(maxlen=180)
        self.y_history: deque[float] = deque(maxlen=180)
        self.speed_history: deque[float] = deque(maxlen=180)
        self.preview_image = None

        self._build_ui()
        self.root.after(50, self.process_queue)

    def _build_ui(self) -> None:
        header = tk.Frame(self.root)
        header.pack(fill="x", padx=16, pady=12)

        tk.Button(header, text="Select Video", command=self.select_video).pack(side="left")

        self.start_button = tk.Button(
            header,
            text="Start",
            command=self.start_processing,
            state="disabled",
        )
        self.start_button.pack(side="left", padx=8)

        self.stop_button = tk.Button(
            header,
            text="Stop",
            command=self.stop_processing,
            state="disabled",
        )
        self.stop_button.pack(side="left")

        self.file_label = tk.Label(header, text="No video selected", anchor="w")
        self.file_label.pack(side="left", fill="x", expand=True, padx=(12, 0))

        info = tk.Frame(self.root)
        info.pack(fill="x", padx=16)

        self.status_label = tk.Label(info, text="Idle", anchor="w")
        self.status_label.pack(fill="x")

        self.frame_label = tk.Label(info, text="frame: -", anchor="w")
        self.frame_label.pack(fill="x")

        self.value_label = tk.Label(info, text="y: -    speed: -", anchor="w")
        self.value_label.pack(fill="x")

        self.progress = ttk.Progressbar(self.root, maximum=100)
        self.progress.pack(fill="x", padx=16, pady=(8, 12))

        content = tk.PanedWindow(self.root, orient="horizontal", sashrelief="flat")
        content.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        chart_frame = tk.Frame(content)
        preview_frame = tk.Frame(content, bg="#111111")
        content.add(chart_frame, minsize=700)
        content.add(preview_frame, minsize=360)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax_speed = self.figure.add_subplot(111)
        self.ax_y = self.ax_speed.twinx()

        self.speed_line, = self.ax_speed.plot([], [], color="tab:blue", label="speed (px/s)")
        self.y_line, = self.ax_y.plot([], [], color="tab:orange", label="y (px)")

        self.ax_speed.set_title("Left Foot Index: y position and speed")
        self.ax_speed.set_xlabel("Frame")
        self.ax_speed.set_ylabel("Speed (px/s)", color="tab:blue")
        self.ax_y.set_ylabel("Y position (px)", color="tab:orange")
        self.ax_speed.grid(True, alpha=0.3)
        lines = [self.speed_line, self.y_line]
        self.ax_speed.legend(lines, [line.get_label() for line in lines], loc="upper right")

        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        tk.Label(
            preview_frame,
            text="Preview",
            bg="#111111",
            fg="white",
            anchor="w",
            padx=12,
            pady=8,
        ).pack(fill="x")

        self.preview_label = tk.Label(
            preview_frame,
            bg="#111111",
            width=480,
            height=360,
        )
        self.preview_label.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def select_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video to analyze",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return

        self.video_path = path
        self.file_label.config(text=Path(path).name)
        self.status_label.config(text="Video selected")
        self.start_button.config(state="normal")
        self.reset_display()

    def reset_display(self) -> None:
        self.frame_history.clear()
        self.y_history.clear()
        self.speed_history.clear()
        self.progress["value"] = 0
        self.frame_label.config(text="frame: -")
        self.value_label.config(text="y: -    speed: -")
        self.speed_line.set_data([], [])
        self.y_line.set_data([], [])
        self.ax_speed.set_xlim(0, 10)
        self.ax_speed.set_ylim(0, 1)
        self.ax_y.set_ylim(0, 1)
        self.preview_label.config(image="", text="")
        self.preview_image = None
        self.canvas.draw()

    def start_processing(self) -> None:
        if not self.video_path or (
            self.processing_thread is not None and self.processing_thread.is_alive()
        ):
            return

        self.reset_display()
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Processing")

        self.processing_thread = threading.Thread(
            target=self.process_video,
            args=(self.video_path,),
            daemon=True,
        )
        self.processing_thread.start()

    def stop_processing(self) -> None:
        self.stop_event.set()
        self.status_label.config(text="Stopping")

    def on_close(self) -> None:
        self.stop_event.set()
        self.root.destroy()

    def process_video(self, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.message_queue.put({"type": "error", "message": "Could not open video"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        prev_point = None
        frame_index = 0
        mp_pose = mp.solutions.pose

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as pose:
            while not self.stop_event.is_set():
                success, frame = cap.read()
                if not success:
                    break

                frame_index += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                y_value = None
                speed_value = 0.0
                point = None
                preview_frame = frame.copy()

                if results.pose_landmarks:
                    landmark = results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.LEFT_FOOT_INDEX
                    ]
                    point = np.array(
                        [
                            landmark.x * frame.shape[1],
                            landmark.y * frame.shape[0],
                        ],
                        dtype=np.float32,
                    )
                    y_value = float(point[1])

                    if prev_point is not None:
                        pixel_distance = float(np.linalg.norm(point - prev_point))
                        speed_value = pixel_distance * fps

                    px, py = int(point[0]), int(point[1])
                    cv2.circle(preview_frame, (px, py), 16, (0, 0, 255), -1)
                    cv2.putText(
                        preview_frame,
                        f"y={y_value:.1f}px",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        preview_frame,
                        f"speed={speed_value:.1f}px/s",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        preview_frame,
                        "foot not detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 180, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if point is not None:
                    prev_point = point

                preview_png = self.frame_to_png_bytes(preview_frame)
                self.message_queue.put(
                    {
                        "type": "frame",
                        "frame_index": frame_index,
                        "total_frames": total_frames,
                        "y_value": y_value,
                        "speed_value": speed_value,
                        "preview_png": preview_png,
                    }
                )

                time.sleep(1.0 / fps)

        cap.release()

        if self.stop_event.is_set():
            self.message_queue.put({"type": "stopped"})
        else:
            self.message_queue.put({"type": "done"})

    def frame_to_png_bytes(self, frame: np.ndarray) -> bytes | None:
        height, width = frame.shape[:2]
        scale = min(480 / width, 360 / height)
        scale = min(scale, 1.0)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        ok, encoded = cv2.imencode(".png", resized)
        if not ok:
            return None
        return encoded.tobytes()

    def process_queue(self) -> None:
        latest_frame = None

        while True:
            try:
                item = self.message_queue.get_nowait()
            except queue.Empty:
                break

            if item["type"] == "frame":
                latest_frame = item
            elif item["type"] == "error":
                self.status_label.config(text=item["message"])
                self.start_button.config(state="normal" if self.video_path else "disabled")
                self.stop_button.config(state="disabled")
            elif item["type"] == "done":
                self.status_label.config(text="Completed")
                self.start_button.config(state="normal" if self.video_path else "disabled")
                self.stop_button.config(state="disabled")
            elif item["type"] == "stopped":
                self.status_label.config(text="Stopped")
                self.start_button.config(state="normal" if self.video_path else "disabled")
                self.stop_button.config(state="disabled")

        if latest_frame is not None:
            self.update_display(latest_frame)

        self.root.after(50, self.process_queue)

    def update_display(self, item: dict) -> None:
        frame_index = item["frame_index"]
        total_frames = item["total_frames"]
        y_value = item["y_value"]
        speed_value = item["speed_value"]
        preview_png = item["preview_png"]

        if y_value is not None:
            self.frame_history.append(frame_index)
            self.y_history.append(y_value)
            self.speed_history.append(speed_value)

        if total_frames > 0:
            self.progress["value"] = frame_index / total_frames * 100

        self.frame_label.config(text=f"frame: {frame_index} / {total_frames}")
        if y_value is None:
            self.value_label.config(text="y: not detected    speed: -")
        else:
            self.value_label.config(
                text=f"y: {y_value:.1f} px    speed: {speed_value:.1f} px/s"
            )

        if self.frame_history:
            frames = list(self.frame_history)
            speeds = list(self.speed_history)
            ys = list(self.y_history)

            self.speed_line.set_data(frames, speeds)
            self.y_line.set_data(frames, ys)

            self.ax_speed.set_xlim(frames[0], max(frames[-1], frames[0] + 1))
            self.ax_speed.set_ylim(0, max(speeds) * 1.1 if max(speeds) > 0 else 1)

            min_y = min(ys)
            max_y = max(ys)
            if min_y == max_y:
                min_y -= 1
                max_y += 1
            margin = (max_y - min_y) * 0.1
            self.ax_y.set_ylim(min_y - margin, max_y + margin)
            self.canvas.draw()

        if preview_png is not None:
            self.preview_image = tk.PhotoImage(data=preview_png)
            self.preview_label.config(image=self.preview_image)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    VideoDualAxisApp().run()
