import torch
import subprocess


def detect_device():
    """ ✅ CUDA 설정 유지 """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            print(f"CUDA is available with {device_count} GPU(s).")
            device_name = torch.cuda.get_device_name(0)
            print(f"Using device: cuda:0 ({device_name})")
        else:
            raise AssertionError("No valid GPU devices found.")
    else:
        raise RuntimeError("CUDA is not available. Ensure you have a compatible GPU installed.")

if __name__ == '__main__':
    # 디바이스 자동 탐지 (출력만 하고 실제 인자는 제거)
    detect_device()

    # 최적의 하이퍼파라미터 설정 (YOLOv7 권장값 기반, 실제 실험에 따라 조정 가능)
    weights = 'yolov7.pt'
    cfg = 'cfg/training/yolov7.yaml'
    data = 'data/forklift.yaml'
    hyp = 'data/hyp.scratch.p5.yaml'
    epochs = 100
    batch_size = 16
    img_size = 640
    name = 'forklift_training'

    # 한 줄로 된 명령어로 변경 (Windows 호환)
    command = (
        f"python train.py --weights {weights} --cfg {cfg} --data {data} --hyp {hyp} "
        f"--epochs {epochs} --batch-size {batch_size} --img-size {img_size} {img_size} "
        f"--name {name} --project runs/train --exist-ok"
    )

    print("[INFO] Training command:")
    print(command)
    subprocess.run(command, shell=True)
