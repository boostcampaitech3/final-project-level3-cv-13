import pyautogui  # 마우스 조작
import cv2
import onnxruntime
import numpy as np
import time
from typing import List, Tuple, Optional

from functions import preprocessing, keypoints_from_heatmaps
# from kpts_inference import OnnxHandModule
from hand_gesture import GestureRecognition
# from cursor_control import Cursor


# kpts_model = OnnxHandModule("../models/m3_cv7a_224x224_220605.onnx")
gesture_recognition = GestureRecognition("../../models/gesture_classification.onnx")


ort_session = onnxruntime.InferenceSession("../../models/m3_cv7a_224x224_220605.onnx")


scr_w, scr_h = pyautogui.size()  # 모니터 해상도: 가로, 세로
# cam_w, cam_h = 640, 480  # 캠에서 읽어올 영상의 크기

RESIZE_WIDTH_HEIGHT = (597, 336)
PATCH_X_MIN, PATCH_X_MAX, PATCH_Y_MIN, PATCH_Y_MAX = 187-120, 411-120, 112, 336

c = np.array([112., 112.], dtype=np.float32)
c = np.expand_dims(c, axis=0)
s = np.array([1.12, 1.12], dtype=np.float32)
s = np.expand_dims(s, axis=0)

# 캠 설정
capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)


# BGR
white = (255, 255, 255)
green = (88, 209, 48)
red = (58, 69, 255)
blue = (255, 132, 10)
purple = (242, 90, 191)
orange = (10, 159, 255)


IDX2GESTURE = {
    0: 'mouse_up',
    1: 'mouse_down',
    2: 'right_click',
    3: None,
    4: None,
}

def get_gesture(gesture: str, prev_gesture: str) -> Optional[str]:
    """이전 동작을 고려하여 수행할 동작 반환"""
    global t_down

    if gesture in (None, 'right_click'):
        return gesture

    if gesture == 'mouse_up' and prev_gesture in ('mouse_down', 'drag', 'wait'):
        t_down = time.time()
        return 'mouse_up'

    elif gesture == 'mouse_down':
        if prev_gesture not in ['mouse_down', 'drag', 'wait']:
            t_down = time.time()
            return 'mouse_down'
        elif time.time() - t_down >= 1.:  # 버튼이 눌린 상태로 1초 유지 -> 드래그
            return 'drag'
        else:
            return 'wait'

    return 'move'

prev_gesture = None
t_down = None

def control(pos: Tuple[int, int], gesture: Optional[str]):
    """마우스 조작"""
    if gesture == 'move' or gesture == 'drag':
        pyautogui.moveTo(pos, _pause=False)
    elif gesture == 'mouse_down':
        pyautogui.mouseDown(_pause=False)
    elif gesture == 'mouse_up':
        pyautogui.mouseUp(_pause=False)
    elif gesture == 'right_click':
        pyautogui.rightClick(_pause=False)
    else:  # None
        pass


if capture.isOpened():

    ret, frame = capture.read()
    while ret and cv2.waitKey(1) < 0:  # 1 ms 마다 캠 읽어오기 & 임의의 키를 누르면 while문 탈출
        fps_counter = time.time()

        ret, frame = capture.read()  # 이미지 읽기 / ret: 성공하면 True, frame: 이미지
        frame = cv2.resize(frame, RESIZE_WIDTH_HEIGHT)

        patch = frame[PATCH_Y_MIN: PATCH_Y_MAX, PATCH_X_MIN: PATCH_X_MAX]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch.flags.writeable = False

        cv2.rectangle(frame, (PATCH_X_MIN, PATCH_Y_MIN), (PATCH_X_MAX, PATCH_Y_MAX-2), (255,255,255), 3)

        
        patch = preprocessing(patch)
        ort_inputs = {ort_session.get_inputs()[0].name: patch}
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_result = keypoints_from_heatmaps(heatmaps=ort_outs[0], center=c, scale=s)
        
        
        # print(onnx_result[0][0].shape) # (21, 2) xy coordinate
        # print(onnx_result[0][0]) # 0-224 float quantized by 1
        # print(onnx_result[1][0].shape) # (21, 1) confidence
        # print(onnx_result[1][0]) # 0-1 float       
        

        # 검지손가락의 끝마디의 threshold
        if onnx_result[1][0][8][0] > 0.5:



            hand_pose = gesture_recognition(onnx_result[0][0])
            # print(hand_pose)
            if hand_pose == None:
                color = white
            if hand_pose == 0:
                color = green
            if hand_pose == 1:
                color = red
            if hand_pose == 2:
                color = blue
            if hand_pose == 3:
                color = purple
            if hand_pose == 4:
                color = orange




            finger_xy = onnx_result[0][0]
            finger_xy = np.array(finger_xy)
            finger_xy = finger_xy + (PATCH_X_MIN, PATCH_Y_MIN)
            finger_xy = finger_xy.tolist()

            for i in range(21):
                cv2.circle(frame, (int(finger_xy[i][0]), int(finger_xy[i][1])), 1, color, 2)   

            keypoints_sequence = ((0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16), (13,17), (17,18), (18,19), (19,20), (0,17))
            for start, end in keypoints_sequence:
                cv2.line(frame, (int(finger_xy[start][0]), int(finger_xy[start][1])), (int(finger_xy[end][0]), int(finger_xy[end][1])), color, 1)
            
            cv2.circle(frame, (int(finger_xy[8][0]), int(finger_xy[8][1])), 10, color, 2) 




            cursor_x, cursor_y = int((1-onnx_result[0][0][8][0]/224) * scr_w), int(onnx_result[0][0][8][1]/224 * scr_h)  # 전체 화면에 대한 좌표 계산
            
            if cursor_x < scr_w/2:
                cursor_x = cursor_x - ((scr_w/2) - cursor_x) * 0.5
            else:
                cursor_x = cursor_x + (cursor_x - (scr_w/2)) * 0.5
            if cursor_y < scr_h/2:
                cursor_y = cursor_y - ((scr_h/2) - cursor_y) * 0.15
            else:
                cursor_y = cursor_y + (cursor_y - (scr_h/2)) * 1.15


            pyautogui.moveTo(cursor_x, cursor_y, _pause=False)  # 마우스 커서 이동

            # if hand_pose is not None: 
            #     gesture = get_gesture(IDX2GESTURE[hand_pose], prev_gesture)
            #     print(cursor_x, cursor_y)
            #     print(gesture)
            #     control((cursor_x, cursor_y), gesture)
            #     prev_gesture = gesture




        time_taken = time.time() - fps_counter
        fps = int(1 / time_taken)
        # print(fps)
        frame_flipped = cv2.flip(frame, 1)
        cv2.putText(frame_flipped, 'PLACE YOUR HAND INSIDE THE BOX', (RESIZE_WIDTH_HEIGHT[0]-PATCH_X_MAX-5, PATCH_Y_MIN-9), cv2.FONT_HERSHEY_SIMPLEX, 0.413, (255,255,255), 1)
        cv2.putText(frame_flipped, f'FPS: {fps}', (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.413, (255,255,255), 1)
        cv2.imshow("Cursor Control", frame_flipped)  # 이미지 출력

    capture.release()
    cv2.destroyAllWindows()
