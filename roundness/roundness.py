from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像ファイルの読み込み
img1_path = "kaiseki1.png"
img2_path = "kaiseki2.png"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 赤い円の検出のため、赤チャンネルの差分でマスクを作成
def extract_red_contour(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 赤色の範囲（2つに分かれる）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 真円度の計算（最大輪郭に対して）
def calculate_circularity(contours):
    if not contours:
        return 0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return circularity

# 輪郭抽出と真円度計算
contours1 = extract_red_contour(img1)
contours2 = extract_red_contour(img2)

circularity1 = calculate_circularity(contours1)
circularity2 = calculate_circularity(contours2)

circularity1, circularity2
