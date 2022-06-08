"""
SECONDS_TO_RECORD를 설정하세요
FPS를 설정하세요
ANNOTATOR_ID는 캠퍼 아이디 마지막 숫자 두 개 입니다
cv7ahand/training/rgb 폴더에 이미지들을 저장합니다
"""

import cv2
import time
from datetime import datetime
import os
from copy import deepcopy

SECONDS_TO_RECORD = 60
FPS = 3
ANNOTATOR_ID = "49" # 김재훈_T3049, 송진우_T3114, 이종민_T3165, 조정빈_T3209, 천영호_T3216
RESIZE_WIDTH_HEIGHT = (568, 336) #  16:9, 224x224의 1.5배
PATCH_X_MIN, PATCH_X_MAX, PATCH_Y_MIN, PATCH_Y_MAX = 172-120, 396-120, 112, 336

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
    frame_ = deepcopy(frame)
    patch = frame_[PATCH_Y_MIN:PATCH_Y_MAX, PATCH_X_MIN:PATCH_X_MAX]
    cv2.rectangle(frame, (PATCH_X_MIN, PATCH_Y_MIN), (PATCH_X_MAX, PATCH_Y_MAX-2), (255,255,255), 3)
    cv2.putText(frame, 'PLACE YOUR HAND INSIDE THE BOX', (PATCH_X_MIN-1, PATCH_Y_MIN-9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.imshow("frame", frame)

    if time.time() - fps_reset_time >= 1 / FPS:
        fps_reset_time = time.time()

        now = datetime.now()
        img_time = now.strftime("%y%m%d%H%M%S%f")
        img_time = img_time[:14]
        
        cv2.imwrite(os.path.join(img_save_path, f"{ANNOTATOR_ID}{img_time}.jpg"), patch)

    if time.time() - start_recording_time >= SECONDS_TO_RECORD:
        break

    if cv2.waitKey(1) == ord('q'):
        break

# writer.release() # video
cap.release()
cv2.destroyAllWindows()
