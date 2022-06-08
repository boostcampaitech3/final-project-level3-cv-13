import os
import argparse

import cv2
import pyautogui

from kpts_inference import MediapipeHandModule, OnnxHandModule
from hand_gesture import GestureRecognition
from cursor_control import Cursor


def float_within_bound(min_, max_):
    def inner_func(x):
        x = float(x)
        if not (min_ <= x < max_):
            raise argparse.ArgumentError(f'invalid value / min: {min_}, max: {max_}')
        return x
    return inner_func


parser = argparse.ArgumentParser()
parser.add_argument(
    '--cam_wh', nargs='+', type=int, default=[960, 540],
    help='웹켐 소스 크기')
parser.add_argument(
    '--infer_xywh', nargs='+', type=int, default=[0, 270, 400, 270],
    help='손동작 인식 영역')
parser.add_argument(
    '--ctr_area_ratio', type=float_within_bound(.5, 1.), default=.6,
    help='손동작 인식 영역 중 커서 조작 영역의 비율'
)
parser.add_argument(
    '--cam_delay', type=int, default=50,
    help='마우스 조작간 딜레이(ms단위)'
)
parser.add_argument(
    '--kpts_model_path', type=str, default=None,
    help='키포인트 모델 경로, None인 경우 mediapipe 사용'
)
parser.add_argument(
    '--gesture_model_path', type=str, default='./prototype/hand_gesture_classification/gesture_classification.onnx',
    help='손동작 분류 모델 경로'
)
args = parser.parse_args()


def main(args):
    # :+:+:+: 인퍼런스 영역, 커서 조작 영역 설정
    if len(args.cam_wh) == 2:
        cam_w, cam_h = args.cam_wh  # 캠 소스 크기
    else:
        raise ValueError(f'cam_wh: 가로, 세로 크기를 입력하세요. / 입력: {args.cam_wh}')

    # 인퍼런스 영역 좌표
    if len(args.infer_xywh) == 4:
        infer_x_min, infer_y_min, infer_w, infer_h = args.infer_xywh
        infer_x_max, infer_y_max = infer_x_min + infer_w, infer_y_min + infer_h
    else:
        raise ValueError(f'infer_xywh: 인퍼런스 영역의 좌상단 좌표와 가로, 세로를 입력하세요. / 입력: {args.infer_xywh}')

    # 커서 조작 영역 좌표
    reduce_ratio = (1. - args.ctr_area_ratio) / 2.
    ctr_x_min, ctr_x_max = int(infer_x_min + reduce_ratio * infer_w), int(infer_x_max - reduce_ratio * infer_w)
    # ctr_y_min, ctr_y_max = int(infer_y_min + reduce_ratio * infer_h), int(infer_y_max - reduce_ratio * infer_h)
    ctr_y_min, ctr_y_max = infer_y_min, int(infer_y_max - reduce_ratio * infer_h * 2.)
    print(infer_x_min, infer_x_max, infer_y_min, infer_y_max)
    print(ctr_x_min, ctr_x_max, ctr_y_min, ctr_y_max)

    # :+:+:+: 웹캠 설정
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    # :+:+:+: 인퍼런스 모델 설정
    if args.kpts_model_path is not None:
        kpts_model = OnnxHandModule(args.kpts_model_path)
    else:
        kpts_model = MediapipeHandModule()

    # :+:+:+: 손동작 분류 모델 설정
    gesture_recognition = GestureRecognition(args.gesture_model_path)

    # :+:+:+: 커서 조작 설정
    cursor = Cursor(args.ctr_area_ratio)

    # :+:+:+: 메인 코드
    if capture.isOpened():
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            key = cv2.waitKey(args.cam_delay)
            if key == ord('q'):  # q 입력시 종료 / 현재 우분투에서는 일부 키 제외하고 segmentation fault 발생
                break

            patch = frame[infer_y_min: infer_y_max+1, infer_x_min: infer_x_max+1]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch.flags.writeable = False

            # 키포인트 예측
            kpts = kpts_model(patch)

            if kpts:
                # 손동작 분류
                hand_pose = gesture_recognition(kpts)
                # 마우스 조작
                cursor(kpts, hand_pose)

            cv2.rectangle(frame, (infer_x_min, infer_y_min), (infer_x_max, infer_y_max), (128, 0, 255), 3)
            cv2.rectangle(frame, (ctr_x_min, ctr_y_min), (ctr_x_max, ctr_y_max), (255, 0, 128), 3)
            cv2.imshow('ho', cv2.flip(frame, 1))
        capture.release()


if __name__ == '__main__':
    main(args)
