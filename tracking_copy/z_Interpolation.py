import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.interpolate import interp1d

# --- 설정값 ---
T_obs = 8          # 관측 길이 (frames)
T_pred = 12        # 예측 길이 (frames)
fps = 5            # 영상 FPS (시각적 안정감 목적)
trail_length = 60  # 과거 궤적 시각화 길이
dot_color = 'red'  # 예측 끝점 색상 (강조용)

# --- 데이터 로딩 ---
df = pd.read_csv("runs/detect/exp29/tracking_ground_coordinates_smoothed.csv") # 스무딩된 파일 사용
df = df[(df["class_id"] == 1) &
        (df["gx"] >= 0.0) & (df["gx"] <= 8.0) &
        (df["gy"] >= 0.0) & (df["gy"] <= 7.0)]

frame_max = df["frame"].max()
ids = df["id"].unique()
colors = {id: np.random.rand(3,) for id in ids}

# --- 시각화 초기화 ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Forklift Trajectory with STGCNN-style Prediction")
ax.grid(True)

lines = {id: ax.plot([], [], '-', color=colors[id])[0] for id in ids}
predict_lines = {id: ax.plot([], [], '--', color=colors[id], alpha=0.5)[0] for id in ids}
pred_dots = {id: ax.plot([], [], 'o', color=dot_color, markersize=6)[0] for id in ids}
texts = {id: ax.text(0, 0, '', fontsize=10, color=colors[id]) for id in ids}

# --- 프레임별 업데이트 함수 ---
def update(frame):
    for id in ids:
        obj = df[(df["id"] == id) & (df["frame"] <= frame)]
        obj_trail = obj[obj["frame"] >= frame - trail_length]

        # 실측 궤적 (실선)
        if not obj_trail.empty:
            x_trail = obj_trail["gx"].tolist()
            y_trail = obj_trail["gy"].tolist()
            lines[id].set_data(x_trail, y_trail)
            last_x, last_y = x_trail[-1], y_trail[-1]
            texts[id].set_position((last_x + 0.1, last_y + 0.1))
            texts[id].set_text(f"forklift_{id}")
        else:
            lines[id].set_data([], [])
            texts[id].set_text("")

        # 예측 경로 (점선) + 끝점 불릿
        if len(obj) >= T_obs:
            obj_sorted = obj.sort_values(by="frame").iloc[-T_obs:]
            frames = obj_sorted["frame"].to_numpy()
            gx = obj_sorted["gx"].to_numpy()
            gy = obj_sorted["gy"].to_numpy()

            try:
                fx = interp1d(frames, gx, kind='linear', fill_value='extrapolate')
                fy = interp1d(frames, gy, kind='linear', fill_value='extrapolate')
                future_frames = np.arange(frames[-1] + 1, frames[-1] + 1 + T_pred)
                pred_x = fx(future_frames)
                pred_y = fy(future_frames)
                predict_lines[id].set_data(pred_x, pred_y)
                pred_dots[id].set_data([pred_x[-1]], [pred_y[-1]])  # 불릿 마커 (강조용)
            except:
                predict_lines[id].set_data([], [])
                pred_dots[id].set_data([], [])
        else:
            predict_lines[id].set_data([], [])
            pred_dots[id].set_data([], [])

    return (
        list(lines.values()) +
        list(predict_lines.values()) +
        list(pred_dots.values()) +
        list(texts.values())
    )

# --- 애니메이션 저장 ---
ani = animation.FuncAnimation(
    fig, update,
    frames=range(0, frame_max + 1),
    interval=1000 / fps,
    blit=True
)

ani.save("runs/detect/exp29/trajectory_predicted.mp4", writer='ffmpeg', fps=fps)
plt.close()
print("Video saved successfully: trajectory_predicted.mp4") 

