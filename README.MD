## 모델 다운로드

아래 링크에서 학습된 YOLOv7 모델 가중치를 다운로드하세요:

- `epoch_079.pt`: [Google Drive 링크](https://drive.google.com/file/d/1SqmHp7Onefwhzj-2Yq_LN0puTlV_Cr0A/view?usp=sharing)

다운로드한 파일은 `yolov7/`, `tracking/` 등 각 디렉토리에 적절히 배치하세요.

---

## 클래스 정의

| 클래스 ID | 클래스 이름    | 설명                                       |
|------------|----------------|--------------------------------------------|
| 0          | person          | 일반 작업자 (WO-01, WO-02 포함)            |
| 1          | forklift        | 지게차 (WO-04)                             |
| 2          | UA-10           | 지게차 이동통로 내 작업자 존재 상황        |
| 3          | UA-12           | 포크 위 탑승 금지 미준수 상황              |
| 4          | UA-13           | 화물 적재 불량 및 붕괴 위험                |
| 5          | helmet          | 헬멧 착용 여부                             |
| 6          | falling         | 작업자 낙상 또는 누운 상태                |

---

## 추적 실행 방법 (DeepSORT 기반)

`tracking/` 디렉토리에서 다음 명령어를 실행하면 추적을 시작할 수 있습니다:

python deep_sort_tracking_id2.py --weights epoch_079.pt --source crop.mp4 --names data/forklift.names --classes 0 1


- `--weights`: 사용하려는 YOLOv7 모델 가중치 (`.pt` 파일)
- `--source`: 추적 대상 영상 파일 경로
- `--names`: 클래스 이름 파일 (한 줄당 하나의 클래스 이름 포함)
- `--classes 0 1`: 사람(0), 지게차(1)만 탐지 및 추적

---

## 참고 사항

- `.pt`, `.pkl`, `.mp4` 등 대용량 파일은 `.gitignore`에 의해 GitHub에 포함되지 않습니다.
- 해당 파일은 Google Drive 등 외부 링크를 통해 다운로드하여 수동으로 배치해 주세요.
- 실시간 처리를 위해 GPU(CUDA 환경)에서 실행하는 것을 권장합니다.

---

