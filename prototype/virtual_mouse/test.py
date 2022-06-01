import pyautogui  # 마우스 조작
import cv2
import mediapipe as mp  # 손 키포인트 디텍션
from prototype.virtual_mouse.hand_gesture import GestureRecognition

sc_w, sc_h = pyautogui.size()  # 모니터 해상도: 가로, 세로
cam_w, cam_h = 640, 480  # 캠에서 읽어올 영상의 크기

infer_x_min, infer_x_max = 0, 420  # 캠 영상에서 인퍼런스할 영역
infer_y_min, infer_y_max = 250, 490 ## 지정한 영역만 잘라서 인퍼런스

infer_w = infer_x_max - infer_x_min
infer_h = infer_y_max - infer_y_min

ctr_size_ratio = .2
ctr_x_min, ctr_x_max = int(infer_x_min + ctr_size_ratio * infer_w), int(infer_x_max - ctr_size_ratio * infer_w)
ctr_y_min, ctr_y_max = int(infer_y_min + ctr_size_ratio * infer_h), int(infer_y_max - ctr_size_ratio * infer_h)

mp_hands = mp.solutions.hands

# 캠 설정
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

gr = GestureRecognition(ctr_size_ratio)

with mp_hands.Hands(  # 디텍션 모델 설정
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    if capture.isOpened():

        ret, frame = capture.read()
        while ret and cv2.waitKey(30) < 0:  # 5 ms 마다 캠 읽어오기 & 임의의 키를 누르면 while문 탈출
            ret, frame = capture.read()  # 이미지 읽기 / ret: 성공하면 True, frame: 이미지

            # 일부 영역만 잘라서 inference
            infer_patch = frame[infer_y_min: infer_y_max, infer_x_min: infer_x_max]
            infer_patch = cv2.cvtColor(infer_patch, cv2.COLOR_BGR2RGB)
            infer_patch.flags.writeable = False
            results = hands.process(infer_patch)

            # inference 영역 표시
            cv2.rectangle(frame, (infer_x_min, infer_y_min), (infer_x_max, infer_y_max), (0, 0, 128), 3)
            cv2.rectangle(frame, (ctr_x_min, ctr_y_min), (ctr_x_max, ctr_y_max), (0, 128, 0), 3)

            # 여러개의 손을 인식함. 먼저 첫번째 손만 사용
            fig_xy, gesture, cursor_pos, valid = gr(results)
            if cursor_pos is not None:
                idx_x, idx_y = fig_xy.idx_t
                idx_f_x, idx_f_y = int(infer_x_min + idx_x * infer_w), int(infer_y_min + idx_y * infer_h)

                m_x, m_y = int((1 - cursor_pos[0]) * sc_w), int(cursor_pos[1] * sc_h)  # 전체 화면에 대한 좌표 계산
                if gesture in ('move', 'drag') and valid:
                    pyautogui.moveTo(m_x, m_y, _pause=False)  # 마우스 커서 이동
                    gesture_color = (255, 0, 128) if gesture == 'move' else (128, 0, 255)
                elif gesture == 'mouse_down':
                    pyautogui.mouseDown(_pause=False)
                    gesture_color = (0, 0, 255)
                elif gesture == 'mouse_up':
                    pyautogui.mouseUp(_pause=False)
                    gesture_color = (255, 0, 0)
                else:
                    gesture_color = (0, 0, 0)
                cv2.circle(frame, (idx_f_x, idx_f_y), 5, gesture_color, -1)
            cv2.imshow("VideoFrame", frame)  # 이미지 출력

        capture.release()
        cv2.destroyAllWindows()
