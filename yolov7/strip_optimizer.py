import os
import json

# 해상도
IMG_W, IMG_H = 1920, 1080

# 딱 이 클래스만 유지
CLASS_MAP = {
    'person': 0,        # 실제로는 'WO-01' 또는 'WO-02'로 맵핑될 수도 있음
    'forklift': 1,      # 실제로는 'WO-04'
    'UA-10': 2,
    'UA-12': 3,
    'UA-13': 4,
    'UA-14': 5
}

# 클래스 ID 변환: AI Hub → YOLO 기준 클래스 이름으로 맵핑
ID_TO_CLASS = {
    'WO-01': 'person',
    'WO-02': 'person',
    'WO-04': 'forklift',
    'UA-10': 'UA-10',
    'UA-12': 'UA-12',
    'UA-13': 'UA-13',
    'UA-14': 'UA-14'
}

# 디렉토리 설정
json_dir = r'C:\eureka_forklift\yolov7\danger_data\labels'
txt_out_dir = r'C:\eureka_forklift\yolov7\danger_data\labels_converted'
os.makedirs(txt_out_dir, exist_ok=True)

converted, skipped = 0, 0

for fname in os.listdir(json_dir):
    if not fname.endswith('.json'):
        continue

    fpath = os.path.join(json_dir, fname)
    with open(fpath, encoding='utf-8') as f:
        data = json.load(f)

    anns = data.get('Learning data info.', {}).get('annotation', [])
    yolo_lines = []

    for ann in anns:
        raw_id = ann['class_id']
        if raw_id not in ID_TO_CLASS:
            skipped += 1
            continue
        model_cls = ID_TO_CLASS[raw_id]
        cls_id = CLASS_MAP[model_cls]

        x, y, w, h = ann['coord']
        xc = (x + w / 2) / IMG_W
        yc = (y + h / 2) / IMG_H
        ww = w / IMG_W
        hh = h / IMG_H

        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

    # 저장
    out_name = os.path.splitext(fname)[0] + '.txt'
    out_path = os.path.join(txt_out_dir, out_name)
    with open(out_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    converted += 1

print(f"✅ 변환 완료: {converted}개 JSON 파일 처리")
print(f"🚫 무시된 객체 수 (필요 없는 클래스): {skipped}")
