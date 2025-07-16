# flowchain_density_video.py
# Flowchain 예측 기반 density map + 중심점 + 객체 간 거리 계산 및 위험도 판단

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import multivariate_normal
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from itertools import combinations

# Flowchain 스타일 예측 모델 정의
class FlowchainPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, num_layers=2):
        super(FlowchainPredictor, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_to_mean = nn.Linear(hidden_dim, output_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs_traj, pred_len=12):
        batch_size = obs_traj.size(0)
        _, (h_n, c_n) = self.encoder(obs_traj)
        decoder_input = obs_traj[:, -1:, :]
        outputs_mean, outputs_logvar = [], []
        h, c = h_n, c_n
        for _ in range(pred_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            mean = self.hidden_to_mean(out.squeeze(1))
            logvar = self.hidden_to_logvar(out.squeeze(1))
            decoder_input = mean.unsqueeze(1)
            outputs_mean.append(mean)
            outputs_logvar.append(logvar)
        pred_means = torch.stack(outputs_mean, dim=1)
        pred_logvars = torch.stack(outputs_logvar, dim=1)
        return pred_means, pred_logvars

obs_len = 8
pred_len = 12
fps = 30

resolution = 0.1
x = np.arange(0, 8 + resolution, resolution)
y = np.arange(0, 7 + resolution, resolution)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

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

model = FlowchainPredictor()
model.eval()

df = pd.read_csv("runs/detect/exp28/tracking_output.csv")
df.columns = df.columns.str.strip()
print("✅ CSV 로드 완료. 총 행 수:", len(df))

df = df[df["class_id"] == 1].copy()
print("✅ 지게차(class_id==1) 필터링 후 행 수:", len(df))
print("고유 ID 수:", df["id"].nunique())

df[["gx", "gy"]] = df.apply(lambda row: pd.Series(pixel_to_ground(row["c_x"], row["c_y"])), axis=1)

id_list = [id for id in df["id"].unique() if len(df[df["id"] == id]) >= obs_len]
print("✅ 예측 가능한 ID 수:", len(id_list))

predictions = {}
for obj_id in id_list:
    obj_df = df[df["id"] == obj_id].sort_values("frame")
    obs_traj = obj_df.iloc[-obs_len:][["gx", "gy"]].values
    obs_tensor = torch.tensor(obs_traj, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean, logvar = model(obs_tensor, pred_len=pred_len)
    mean = mean.squeeze(0).numpy()
    std = np.exp(0.5 * logvar.squeeze(0).numpy())
    predictions[obj_id] = list(zip(mean, std))

print("✅ 예측 완료된 객체 수:", len(predictions))

if not predictions:
    print("⚠️ 예측 가능한 지게차 객체가 없습니다. CSV 데이터를 확인하세요.")
    exit()

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Flowchain Trajectory Density")
ax.grid(True)

# 애니메이션 프레임 생성 함수
def update(frame):
    ax.clear()
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Flowchain Trajectory Density (T+{frame+1})")
    ax.grid(True)
    total_density = np.zeros_like(X)
    centers = {}

    for obj_id, pred_list in predictions.items():
        if frame >= len(pred_list):
            continue
        mu, std = pred_list[frame]
        cov = np.diag(std ** 2)
        rv = multivariate_normal(mu, cov)
        density = rv.pdf(pos)
        total_density += density
        centers[obj_id] = mu
        ax.plot(mu[0], mu[1], 'o', color='blue')
        ax.text(mu[0]+0.1, mu[1]+0.1, f"id {obj_id}", fontsize=9, color='blue')

    max_val = total_density.max()
    print(f"[frame {frame}] max density: {max_val:.5f}")
    if max_val > 0:
        ax.contourf(X, Y, total_density, cmap='hot', alpha=0.6)

    # 유클리디안 거리 계산 + 위험도 판단
    for (id1, p1), (id2, p2) in combinations(centers.items(), 2):
        dist = np.linalg.norm(p1 - p2)
        if dist < 30:
            risk = "위험"
            color = "red"
        elif dist < 100:
            risk = "주의"
            color = "orange"
        else:
            risk = "안전"
            color = "green"
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='--', color=color, alpha=0.5)
        mid_x, mid_y = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.text(mid_x, mid_y, f"{risk} ({dist:.1f}m)", fontsize=8, color=color)

print("🎬 애니메이션 생성 시작...")
ani = animation.FuncAnimation(fig, update, frames=pred_len, interval=1000/fps, blit=False)
print("💾 저장 시작...")
ani.save("runs/detect/exp28/flowchain_trajectory.mp4", writer='ffmpeg', fps=fps)
plt.close()
print("✅ 영상 저장 완료: flowchain_trajectory.mp4")
