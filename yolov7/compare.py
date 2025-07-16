import os
from pathlib import Path
from collections import defaultdict
import csv

# 설정
IMG_W, IMG_H = 1920, 1080
IOU_THRESHOLD = 0.5

# 클래스 이름 매핑
CLASS_NAMES = {
    0: 'person (WO-01)',
    1: 'forklift (WO-04)',
    2: 'UA-10',
    3: 'UA-12',
    4: 'UA-13',
    5: 'helmet',
    6: 'falling'
}

# IOU 계산
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

# YOLO → [x1, y1, x2, y2]
def yolo_to_xyxy(x, y, w, h):
    x1 = (x - w / 2) * IMG_W
    y1 = (y - h / 2) * IMG_H
    x2 = (x + w / 2) * IMG_W
    y2 = (y + h / 2) * IMG_H
    return [x1, y1, x2, y2]

# 경로 설정
gt_dir = Path(r'C:\eureka_forklift\yolov7\danger_data\labels_converted')
pred_dir = Path(r'C:\eureka_forklift\yolov7\runs\detect\danger_test\labels')
csv_out_path = Path(r'C:\eureka_forklift\yolov7\comparison_result.csv')

# 클래스별 통계
class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

# 비교 수행
for gt_file in gt_dir.glob('*.txt'):
    base = gt_file.stem
    pred_file = pred_dir / f'{base}.txt'
    if not pred_file.exists():
        continue

    # 정답 로딩
    gt_boxes = []
    with open(gt_file) as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split()[:5])
            gt_boxes.append((int(cls), yolo_to_xyxy(x, y, w, h)))

    # 예측 로딩
    pred_boxes = []
    with open(pred_file) as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split()[:5])
            pred_boxes.append((int(cls), yolo_to_xyxy(x, y, w, h)))

    matched_pred = set()

    # GT와 예측 비교 (TP, FN)
    for gt_idx, (gt_cls, gt_box) in enumerate(gt_boxes):
        if gt_cls not in CLASS_NAMES:
            continue
        matched = False
        for pred_idx, (pred_cls, pred_box) in enumerate(pred_boxes):
            if pred_cls == gt_cls and iou(gt_box, pred_box) >= IOU_THRESHOLD and pred_idx not in matched_pred:
                matched = True
                matched_pred.add(pred_idx)
                class_stats[gt_cls]['tp'] += 1
                break
        if not matched:
            class_stats[gt_cls]['fn'] += 1

    # FP 계산 (예측했지만 GT와 매칭되지 않은 것들)
    for pred_idx, (pred_cls, pred_box) in enumerate(pred_boxes):
        if pred_cls not in CLASS_NAMES:
            continue
        if pred_idx not in matched_pred:
            class_stats[pred_cls]['fp'] += 1

# 결과 출력
print("\n📊 클래스별 정탐 성능 (IOU ≥ 0.5 기준)")
print(f"{'Class':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'Recall':>10} {'Precision':>10} {'Accuracy':>10}")
csv_rows = []

for cls_id in sorted(class_stats.keys()):
    name = CLASS_NAMES[cls_id]
    stat = class_stats[cls_id]
    tp, fp, fn = stat['tp'], stat['fp'], stat['fn']
    recall = tp / (tp + fn + 1e-6) * 100
    precision = tp / (tp + fp + 1e-6) * 100
    accuracy = tp / (tp + fn + fp + 1e-6) * 100

    print(f"{name:<20} {tp:>5} {fp:>5} {fn:>5} {recall:>9.2f}% {precision:>10.2f}% {accuracy:>9.2f}%")
    csv_rows.append([cls_id, name, tp, fp, fn, f"{recall:.2f}%", f"{precision:.2f}%", f"{accuracy:.2f}%"])

# CSV 저장
with open(csv_out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Class ID', 'Class Name', 'TP', 'FP', 'FN', 'Recall (%)', 'Precision (%)', 'Accuracy (%)'])
    writer.writerows(csv_rows)

print(f"\n✅ 결과 CSV 저장 완료: {csv_out_path}")
