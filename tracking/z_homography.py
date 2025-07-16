import pandas as pd
import numpy as np
import cv2

# Homography 기준점
pixel_points = np.array([
    [446, 1004],
    [1906, 956],
    [1487, 549],
    [661, 615]
], dtype=np.float32)

ground_points = np.array([
    [0.0, 0.0],
    [8.0, 0.0],
    [8.0, 7.0],
    [0.0, 7.0]
], dtype=np.float32)

H, _ = cv2.findHomography(pixel_points, ground_points)

def pixel_to_ground(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return out[0][0]

# CSV 로드
df = pd.read_csv("runs/detect/exp28/tracking_output.csv")
df.columns = df.columns.str.strip()  # 혹시 모를 공백 제거

# 중심점 -> 지면 좌표로 변환
df[["gx", "gy"]] = df.apply(lambda row: pd.Series(pixel_to_ground(row["c_x"], row["c_y"])), axis=1)

# 저장
df.to_csv("runs/detect/exp28/tracking_ground_coordinates.csv", index=False)
print("✅ 변환 완료: tracking_ground_coordinates.csv 저장됨")
