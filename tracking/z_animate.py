import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# CSV 불러오기
df = pd.read_csv("runs/detect/exp28/tracking_ground_coordinates.csv")

# 지게차만 선택 + 사각형 내부만 유지
df = df[(df["class_id"] == 1) &
        (df["gx"] >= 0.0) & (df["gx"] <= 8.0) &
        (df["gy"] >= 0.0) & (df["gy"] <= 7.0)]

# 설정
fps = 30  # 원본 영상 FPS로 맞춰야 시간 길이 일치
trail_length = 60
frame_max = df["frame"].max()
ids = df["id"].unique()
colors = {id: np.random.rand(3,) for id in ids}

# Figure 설정
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Forklift Trajectory")
ax.grid(True)

# 선과 텍스트 초기화
lines = {id: ax.plot([], [], '-', color=colors[id])[0] for id in ids}
texts = {id: ax.text(0, 0, '', fontsize=10, color=colors[id]) for id in ids}

# 애니메이션 프레임별 업데이트
def update(frame):
    for id in ids:
        obj = df[(df["id"] == id) & (df["frame"] <= frame) & (df["frame"] >= frame - trail_length)]
        if not obj.empty:
            x_trail = obj["gx"].tolist()
            y_trail = obj["gy"].tolist()
            lines[id].set_data(x_trail, y_trail)
            last_x, last_y = x_trail[-1], y_trail[-1]
            texts[id].set_position((last_x + 0.1, last_y + 0.1))
            texts[id].set_text(f"forklift_{id}")
        else:
            lines[id].set_data([], [])
            texts[id].set_text("")
    return list(lines.values()) + list(texts.values())

# 애니메이션 저장
ani = animation.FuncAnimation(
    fig, update, frames=range(0, frame_max + 1), interval=1000/fps, blit=True
)

ani.save("runs/detect/exp28/trajectory.mp4", writer='ffmpeg', fps=fps)
plt.close()
print("✅ 최종 시각화 완료: trajectory.mp4")
