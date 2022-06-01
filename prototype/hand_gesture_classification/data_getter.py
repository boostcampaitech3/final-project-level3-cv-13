"""
original repository: https://github.com/kinivi/hand-gesture-recognition-mediapipe

mediapipe의 hand 모듈을 이용하여 지정한 영역 내에서 손의 키포인트 좌표를 기록하고 json 형식으로 저장.

* annotation 생성
    1. annotation 저장 경로와 손 인식 영역, LABEL 설정
        * LABEL: 기록한 모든 손 동작의 class는 LABEL로 지정됨
    2. 실행: `python3 data_getter.py`
    3. annotation 기록
        * 박스로 표시된 인식 영역 내에서 손 동작을 취함
        * 스페이스 입력: 현재 손동작 기록
        * 엔터 입력: 지금까지 기록한 모든 손동작의 키포인트를 LABEL과 함께 json 파일로 저장

* 제스쳐 인식 모델 테스트
    1. 변수 onnx_path와 전처리 함수 preprocessing을 알맞게 수정
    2. onnx 인퍼런스 과정 코드 수정
        * 현재 코드는 (1, 42) shape 입력을 받아 (1, 2) shape 출력을 내는 형식
"""

import os
import cv2
import json
from copy import deepcopy
import numpy as np
import mediapipe as mp

import onnxruntime as ort

# annotation 저장 경로
DATA_DIR = 'data'
LABEL = 4  # 기록되는 모든 손 동작은 annotation 해당 label이 부여됨
ANNOTATION_NAME = f'annotation_{LABEL}.json'
# 0: mouse up, 1: mouse left down, 2: mouse right click, 3: close, 4: open
CAM_DELAY = 100

# onnx 테스트
test_onnx = True
onnx_path = './simple_model.onnx'

# 데이터 저장 디렉토리 생성
if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    num_img = 0
else:
    num_img = len([x for x in os.listdir(DATA_DIR) if x.split('.')[-1] == 'jpg'])
    pass

num_data = 0

# cam 설정
cam_w, cam_h = 640, 480  # 캠에서 읽어올 영상의 크기
crop_size = (224, 224)  # 손을 인식할 영역 크기 설정
crop_x_min, crop_y_min = 50, 240  # 손을 인식항 영역 위치 설정
crop_x_max = crop_x_min + crop_size[0]
crop_y_max = crop_y_min + crop_size[1]

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

######################################################################
if test_onnx:  # onnx 로
    ort_session = ort.InferenceSession("simple_model.onnx")

    def preprocessing(kpts):
        """keypoints 전처리 - 손목지점을 원점으로 지정, 키포인트 x, y 좌표를 0~1로 정규화"""
        kpts_ = deepcopy(kpts)
        kpts_ = (kpts_ - kpts_.min(axis=0)) / (kpts_.max(axis=0) - kpts_.min(axis=0))
        return kpts_.flatten()
######################################################################드


annotations = {'annotations': []}
with mp.solutions.hands.Hands(model_complexity=1) as hands:
    if capture.isOpened():
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            key = cv2.waitKey(CAM_DELAY)
            # print(key)
            if key == ord('q'):
                break

            patch = frame[crop_y_min: crop_y_max, crop_x_min: crop_x_max]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch.flags.writeable = False

            cv2.rectangle(frame, (crop_x_min, crop_y_min), (crop_x_max, crop_y_max), (0, 0, 128), 3)

            kpts = hands.process(patch)

            kpts_xy = []
            if kpts.multi_hand_landmarks:
                for hand_landmarks in kpts.multi_hand_landmarks:

                    for i in range(21):
                        x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                        kpts_xy.extend([x, y])  # 어노테이션 파일에 기록하기 위해 좌표 획득

                        # 화면에 출력하기 위한 픽셀 좌표로 변환
                        hand_landmarks.landmark[i].x = \
                            (crop_x_min + crop_size[0] * hand_landmarks.landmark[i].x) / cam_w
                        hand_landmarks.landmark[i].y = \
                            (crop_y_min + crop_size[1] * hand_landmarks.landmark[i].y) / cam_h

                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style(),
                    )
                    break  # 간혹 한 손을 좌우 양손으로 두 번 인식하는 경우 발생

            ######################################################################
            if test_onnx:  # onnx 인퍼런스
                def softmax(x):
                    x_exp = np.exp(x)
                    denom = np.sum(x_exp)
                    prob = x_exp / denom
                    return prob

                if len(kpts_xy) == 42:
                    kpts_ = preprocessing(np.array(kpts_xy).reshape(-1, 2))[None].astype(np.float32)
                    output = ort_session.run(
                        ['output'],
                        {'input': kpts_}
                    )[0][0]

                    argmax = output.argmax()
                    # prob = softmax(output)
                    # if argmax == 0 and output[argmax] >= .5:
                    print(output)
                    if output[argmax] >= .6:
                        if argmax == 0:
                            print('mouse up')
                        # elif argmax == 1 and output[argmax] >= .5:
                        elif argmax == 1:
                            print('mouse down')
                        elif argmax == 2:
                            print('right click')
                        elif argmax == 3:
                            print('close')
                        elif argmax == 4:
                            print('open')
                    else:
                        print('nothing')

            ######################################################################

            cv2.imshow('ho', cv2.flip(frame, 1))
            # cv2.imshow('ho', frame)

            # 우분투에서 키 입력시 segmentation fault 발생 - 스페이스와 엔터만 이용
            if key == 32:  # space - 손동작 어노테이션 기록
                fname = f'img_{num_img}.jpg'
                cv2.imwrite(os.path.join(DATA_DIR, fname), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                annotations['annotations'].append({
                    # 'fname': fname,
                    'label': LABEL,
                    'kpts': kpts_xy,
                })  # json 형식으로 annotation 기록
                print(num_data)
                num_img += 1
                num_data += 1
            elif key == 13:  # enter - 어노테이션 파일 저장
                anno_path = os.path.join(DATA_DIR, ANNOTATION_NAME)
                with open(anno_path, 'w') as f:
                    json.dump(annotations, f, indent=2)
                print(f'annotations saved. - {anno_path} / label: {LABEL}')
        capture.release()


