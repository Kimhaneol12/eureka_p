## 모델 다운로드

아래 링크에서 학습된 YOLOv7 모델 가중치를 다운로드:

- `epoch_079.pt`: [Google Drive 링크](https://drive.google.com/file/d/1SqmHp7Onefwhzj-2Yq_LN0puTlV_Cr0A/view?usp=sharing)

다운로드한 파일은 `yolov7/`, `tracking/` 등 각 디렉토리에 배치

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

클래스 2, 3, 4는 다음과 같은 데이터셋 사용
- [AI Hub 물류창고 내 안전사고 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EB%AC%BC%EB%A5%98%EC%84%BC&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=510)

아래 링크에서 학습된 YOLOv7 모델 가중치를 다운로드:

- `epoch_079.pt`: [Google Drive 링크](https://drive.google.com/file/d/1SqmHp7Onefwhzj-2Yq_LN0puTlV_Cr0A/view?usp=sharing)

다운로드한 파일은 `yolov7/`, `tracking/` 등 각 디렉토리에 배치

---

## 위험상황 탐지 실행 방법

`yolov7/` 디렉토리에서 추적하고 싶은 mp4, jpg를 배치한 후, 해당 위치에서 다음 명령어를 실행하면 탐지를를 시작:

python detect.py --weights epoch_079.pt --source 3.jpg --img 640 --conf 0.25

---

## 추적 실행 방법 (DeepSORT 기반)

`tracking/` 디렉토리에서 추적하고 싶은 mp4, jpg를 배치한 후, 해당 위치에서 다음 명령어를 실행하면 추적을 시작:

python deep_sort_tracking_id2.py --weights epoch_079.pt --source crop.mp4 --names data/forklift.names --classes 0 1

- `--classes 0 1`: 사람(0), 지게차(1)만 탐지 및 추적

---

## 테스트베드 영상

- [Google Drive 링크](https://drive.google.com/drive/u/0/folders/1GX8vIGMwHYUoBdpJCt2HkeNlqKeMj-2U)

---

## 참고 사항

- `.pt`, `.pkl`, `.mp4` 등 대용량 파일은 `.gitignore`에 의해 GitHub에 미포함
- 해당 파일은 Google Drive 등 외부 링크를 통해 다운로드하여 수동으로 배치 필요
- 실시간 처리를 위해 GPU(CUDA 환경)에서 실행하는 것을 권장

---

