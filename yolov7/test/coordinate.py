import cv2

# 영상 불러오기
video_path = "crop.mp4"
cap = cv2.VideoCapture(video_path)

# 클릭된 좌표 저장용
clicked_points = []

# 클릭 콜백 함수
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"클릭한 좌표: ({x}, {y})")
        clicked_points.append((x, y))

# 특정 프레임 위치로 이동 (예: 첫 프레임)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()
if not ret:
    print("영상을 불러올 수 없습니다.")
    cap.release()
    exit()

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouse_callback)

while True:
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC 눌러 종료
        break

cv2.destroyAllWindows()
cap.release()

print("모든 클릭된 좌표:", clicked_points)
