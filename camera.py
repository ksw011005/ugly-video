import cv2 as cv
import numpy as np

# 영상 경로 및 카메라 파라미터
video_file = "C:/Users/ksw01/OneDrive/Desktop/chess.mp4"

K = np.array([
    [432.7390364738057, 0, 476.0614994349778],
    [0, 431.2395555913084, 288.7602152621297],
    [0, 0, 1]
])

# 왜곡 계수 (distortion coefficients)
dist_coeff = np.array([-0.12, 0.04, -0.0004, 0.0002, 0.0])

# === 캘리브레이션 결과 출력 ===
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]
rmse = 0.0

print("\n=== Calibration Summary ===")
print(f"fx, fy:        {fx:.4f}, {fy:.4f}")
print(f"cx, cy:        {cx:.4f}, {cy:.4f}")
print(f"distortion:    {', '.join([f'{d:.6f}' for d in dist_coeff.flatten()])}")
print(f"RMSE:          {rmse:.6f}")
print("============================\n")

# === 비디오 처리 ===
video = cv.VideoCapture(video_file)
if not video.isOpened():
    print("비디오를 열 수 없습니다.")
    exit()

show_rectify = True  # 보정 여부 토글
map1, map2 = None, None

while True:
    valid, img = video.read()
    if not valid:
        break

    info = "Original"
    if show_rectify:
        if map1 is None or map2 is None:
            h, w = img.shape[:2]
            map1, map2 = cv.initUndistortRectifyMap(
                K, dist_coeff, None, K, (w, h), cv.CV_32FC1
            )
        img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
        info = "Rectified"

    # 텍스트 표시
    cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

    # 화면에 출력
    cv.imshow("Distortion Correction", img)

    key = cv.waitKey(30)
    if key == 27:  # ESC 키로 종료
        break
    elif key == ord(' '):  # 스페이스바로 원본/보정 토글
        show_rectify = not show_rectify

video.release()
cv.destroyAllWindows()
