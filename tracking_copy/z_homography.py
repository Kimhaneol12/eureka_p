import pandas as pd
import numpy as np
import cv2
from scipy.signal import savgol_filter

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
df = pd.read_csv("runs/detect/exp29/tracking_output.csv")
df.columns = df.columns.str.strip()  # 혹시 모를 공백 제거

# 중심점 -> 지면 좌표로 변환
df[["gx", "gy"]] = df.apply(lambda row: pd.Series(pixel_to_ground(row["c_x"], row["c_y"])), axis=1)

# --- Jitter 완화를 위한 Savitzky-Golay 필터 적용 ---
ids = df["id"].unique()
smoothed_df_list = []
for id in ids:
    obj_df = df[df["id"] == id].sort_values(by='frame').copy()
    
    # 데이터가 필터 창 길이보다 길어야 함
    window_length = 15  # 홀수 값으로 조정 가능 (예: 5, 11, 21...)
    polyorder = 2       # window_length보다 작은 값으로 조정 가능
    
    if len(obj_df) > window_length:
        obj_df.loc[:, "gx"] = savgol_filter(obj_df["gx"], window_length, polyorder)
        obj_df.loc[:, "gy"] = savgol_filter(obj_df["gy"], window_length, polyorder)
        
    smoothed_df_list.append(obj_df)

if smoothed_df_list:
    smoothed_df = pd.concat(smoothed_df_list)
    # 스무딩된 데이터프레임을 새로운 CSV 파일로 저장
    smoothed_df.to_csv("runs/detect/exp29/tracking_ground_coordinates_smoothed.csv", index=False)
    print("Smoothed data saved to tracking_ground_coordinates_smoothed.csv")
else:
    # 원본 데이터 저장 (필터링이 적용되지 않은 경우)
    df.to_csv("runs/detect/exp29/tracking_ground_coordinates.csv", index=False)

print("Conversion complete: tracking_ground_coordinates_smoothed.csv saved")
