import pyautogui  # 마우스 조작
import cv2
import mediapipe as mp  # 손 키포인트 디텍션

sc_w, sc_h = pyautogui.size()  # 모니터 해상도: 가로, 세로
cam_w, cam_h = 640, 480  # 캠에서 읽어올 영상의 크기

infer_x_min, infer_x_max = 50, 370  # 캠 영상에서 인퍼런스할 영역
infer_y_min, infer_y_max = 300, 440 ## 지정한 영역만 잘라서 인퍼런스

infer_w = infer_x_max - infer_x_min
infer_h = infer_y_max - infer_y_min

mp_hands = mp.solutions.hands

# 캠 설정
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

with mp_hands.Hands(  # 디텍션 모델 설정
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

  if capture.isOpened():

    ret, frame = capture.read()
    while ret and cv2.waitKey(5) < 0:  # 5 ms 마다 캠 읽어오기 & 임의의 키를 누르면 while문 탈출
        ret, frame = capture.read()  # 이미지 읽기 / ret: 성공하면 True, frame: 이미지

        # 일부 영역만 잘라서 inference
        infer_patch = frame[infer_y_min: infer_y_max, infer_x_min: infer_x_max]
        infer_patch = cv2.cvtColor(infer_patch, cv2.COLOR_BGR2RGB)
        infer_patch.flags.writeable = False
        results = hands.process(infer_patch)

        # inference 영역 표시        
        cv2.rectangle(frame, (infer_x_min, infer_y_min), (infer_x_max, infer_y_max), (0, 0, 128), 3)
        
        # 여러개의 손을 인식함. 먼저 첫번째 손만 사용
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x  # 검지 x
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y  # 검지 y
            # x, y좌표는 0~1로 정규화된 값으로 주어짐.
            
            m_x, m_y = int((1-x) * sc_w), int(y * sc_h)  # 전체 화면에 대한 좌표 계산
            f_x, f_y = int(infer_x_min + x * infer_w), int(infer_y_min + y * infer_h)  # 검지 좌표
            cv2.circle(frame, (f_x, f_y), 5, (255, 0, 128), -1)  # 캠 영상에 위치 표시
            pyautogui.moveTo(m_x, m_y, _pause=False)  # 마우스 커서 이동
        
        cv2.imshow("VideoFrame", frame)  # 이미지 출력

    capture.release()
    cv2.destroyAllWindows()
