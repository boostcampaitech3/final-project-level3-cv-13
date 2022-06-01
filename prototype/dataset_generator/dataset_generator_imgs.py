"""
SECONDS_TO_RECORD 설정하세요
FPS 설정하세요
ANNOTATOR_NAME는 세 글자 이니셜이어야 합니다
RESIZE_WIDTH_HEIGHT는 세로가 224가 되게 16:9로 맞춘 값입니다
cv7ahand/training/rgb 폴더에 이미지를 저장합니다
"""

import cv2
import time
from datetime import datetime
import os

SECONDS_TO_RECORD = 5
FPS = 2
ANNOTATOR_NAME = "KJH"
RESIZE_WIDTH_HEIGHT = (434, 244)

if not os.path.exists("cv7ahand"):
    os.mkdir("cv7ahand")
if not os.path.exists("cv7ahand/training"):
    os.mkdir("cv7ahand/training")
if not os.path.exists("cv7ahand/training/rgb"):
    os.mkdir("cv7ahand/training/rgb")
img_save_path = "cv7ahand/training/rgb"

cap = cv2.VideoCapture(0)

# width, height = int(cap.get(3)), int(cap.get(4)) # video
# writer = cv2.VideoWriter('cv7ahand.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height)) # video

start_recording_time = time.time()
fps_reset_time = time.time()
while True:
    retval, frame = cap.read()
    frame = cv2.resize(frame, RESIZE_WIDTH_HEIGHT)
    # writer.write(frame) # video
    cv2.imshow("frame", frame)

    if time.time() - fps_reset_time >= 1 / FPS:
        fps_reset_time = time.time()

        now = datetime.now()
        img_time = now.strftime("%y%m%d%H%M%S%f")
        img_time = img_time[:14]
        
        cv2.imwrite(os.path.join(img_save_path, f"{ANNOTATOR_NAME}_{img_time}.jpg"), frame)

    if time.time() - start_recording_time >= SECONDS_TO_RECORD:
        break

    if cv2.waitKey(1) == ord('q'):
        break

# writer.release() # video
cap.release()
cv2.destroyAllWindows()
