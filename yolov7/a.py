import torch
from models.experimental import attempt_load

# 모델 경로 (필요시 절대경로로 지정)
model_path = 'epoch_079.pt'
device = 'cpu'  # 또는 'cuda:0'

# 모델 로드
model = attempt_load(model_path, map_location=device)

# 클래스 이름 추출
if hasattr(model, 'names'):
    class_names = model.names
    print(f"class 수: {len(class_names)}")
    print("class 리스트:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")
else:
    print("❌ 클래스 이름 정보를 찾을 수 없습니다.")
