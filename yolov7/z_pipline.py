import cv2
import torch
import numpy as np
import joblib
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device

# 경로 설정
IMAGE_PATH = r'C:\eureka_forklift\yolov7\1.jpg'
MODEL_PATH = r'C:\eureka_forklift\yolov7\epoch_079.pt'
PKL_PATH = r'C:\eureka_forklift\yolov7\fall_detection.pkl'

# 모델 설정
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
CROP_SIZE = (64, 128)  # pkl 모델 학습 시 사용된 이미지 크기

# 장치 설정
device = select_device('')
model = attempt_load(MODEL_PATH, map_location=device)
model.eval()

# 이미지 불러오기
img0 = cv2.imread(IMAGE_PATH)
assert img0 is not None, f"Image not found at {IMAGE_PATH}"

# 전처리
img = letterbox(img0, IMG_SIZE, stride=32, auto=True)[0]
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
img = np.ascontiguousarray(img)
img_tensor = torch.from_numpy(img).to(device).float() / 255.0
if img_tensor.ndimension() == 3:
    img_tensor = img_tensor.unsqueeze(0)

# 추론
with torch.no_grad():
    pred = model(img_tensor, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=CONF_THRES, iou_thres=IOU_THRES)

# pkl 모델 로드
rf_model = joblib.load(PKL_PATH)

# 감지된 사람에 대해 낙상 판단
for det in pred:
    if det is not None and len(det):
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in det:
            cls_id = int(cls)
            if cls_id == 0:  # person only
                x1, y1, x2, y2 = map(int, xyxy)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img0.shape[1], x2)
                y2 = min(img0.shape[0], y2)

                # crop and resize
                crop = img0[y1:y2, x1:x2]
                resized = cv2.resize(crop, CROP_SIZE)
                flattened = resized.flatten().reshape(1, -1)

                # predict
                pred_fall = rf_model.predict(flattened)[0]
                label = 'falling' if pred_fall == 1 else 'normal'
                color = (0, 0, 255) if pred_fall == 1 else (0, 255, 0)

                # draw
                cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img0, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# 결과 출력
cv2.imshow('Fall Detection', img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
