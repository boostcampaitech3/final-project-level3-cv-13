import sys
import argparse
from time import sleep, time
from typing import Tuple, Optional, List

import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import *

from kpts_inference import MediapipeHandModule, OnnxHandModule
from hand_gesture import GestureRecognition
from cursor_control import Cursor


parser = argparse.ArgumentParser()
parser.add_argument(
    '--kpts_model_path', type=str, default='models/m3_cv7a_224x224_220605.onnx',
    help='키포인트 모델 경로, None인 경우 mediapipe 사용'
)
parser.add_argument(
    '--gesture_model_path', type=str, default='models/gesture_classification.onnx',
    help='손동작 분류 모델 경로'
)
args = parser.parse_args()


form_class = uic.loadUiType("gui.ui")[0]


class CursorThread(QThread):
    """
    웹캠을 통해 이미지를 받아 커서 조작과 위젯의 이미지 갱신 수행

    Args:
        p (QObject): 부모 객체, 부모 객체 종료시에 쓰레드도 함께 종료됨.
        time_per_step_label (QLabel): 스텝 당 소요 시간을 표시할 라벨
        img_label (QLabel): 실시간으로 웹캠 이미지를 표시할 라벨

    Attributes:
        change_pixmap (pyqtSignal): 웹캠 이미지 갱신 트리거. 새로운 pixmap을 받으면 이미지 갱신
        delay (float): 부하 조절을 위한 커서 조작 사이의 sleep 시간(초)
        use_left_hand (bool): 왼손잡이 설정 여부
    """
    CAM_W, CAM_H = 640, 480  # 웹캠 소스 크기
    SHOW_W, SHOW_H = 320, 240  # 위젯 이미지 라벨 크기
    CAM_LABEL_RATIO = CAM_W / SHOW_W
    change_pixmap = pyqtSignal(QImage)

    def __init__(self, p: QObject, fps_label: QLabel, img_label: QLabel):
        super().__init__(p)
        self.fps_label = fps_label
        self.img_label = img_label
        self.delay = None
        self.use_left_hand = None

        self.skeleton_color = {
            None: (247, 253, 175),
            0: (248, 184, 179),
            1: (170, 243, 162),
            2: (146, 222, 252),
            3: (174, 197, 241),
            4: (177, 167, 240),
        }

        self.change_pixmap.connect(self.update_img_label)

        self.__box_center = (200, 320)
        self.__box_size = 224

        # :+:+:+: 인퍼런스 모델 설정
        if args.kpts_model_path is not None:
            self.kpts_model = OnnxHandModule(args.kpts_model_path)
        else:
            self.kpts_model = MediapipeHandModule()
        # :+:+:+: 손동작 분류 모델 설정
        self.gesture_recognition = GestureRecognition(args.gesture_model_path)
        # :+:+:+: 커서 조작 설정
        self.cursor = Cursor(None)

    def run(self):
        assert self.delay is not None and self.use_left_hand is not None, f'delay, use_left_hand 설정 필요'

        # 웹캠 설정
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_W)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FPS, 30)

        while True:
            t_1 = time()
            ret, frame = capture.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 인퍼런스 영역 좌표 계산
            l, t = self.__box_center[0] - self.__box_size // 2, self.__box_center[1] - self.__box_size // 2
            r, b = l + self.__box_size, t + self.__box_size

            patch = np.ascontiguousarray(frame[t: b, l: r])
            if self.use_left_hand:
                patch = cv2.flip(patch, 1)
            patch.flags.writeable = False

            # 모델 예측을 이용해 커서 조작
            kpts = self.kpts_model(patch)
            if kpts:
                hand_pose = self.gesture_recognition(kpts)
                if self.use_left_hand:
                    kpts = [x if i % 2 == 1 else 1.-x for i, x in enumerate(kpts)]
                idx_pos = self.cursor(kpts, hand_pose)
                # self.draw_idx_point(frame, (l, t), idx_pos)
                frame = self.draw_skeleton(frame, kpts, (l, t), hand_pose)

            # 인퍼런스 영역과 커서 조작 영역 표시
            frame = self.draw_area(frame, (l, t, r, b))

            # 이미지 갱신
            self.update_img(frame)

            sleep(self.delay)  # 부하 조절
            self.fps_label.setText(f'{1. / (time()-t_1):.02f}')  # 스텝 당 소요시간 업데이트

    def update_img(self, frame: np.ndarray):
        """이미지 라벨 갱신"""
        frame = cv2.flip(frame, 1)  # 보기 편하게 좌우 반전
        h, w, c = frame.shape
        bytes_per_line = w * c
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).scaled(self.SHOW_W, self.SHOW_H)
        self.change_pixmap.emit(q_img)  # pixmap 업데이트 - 위젯 이미지 갱신 트리거

    def update_img_label(self, img: QImage):
        """이미지 라벨의 pixmap 갱신 - update_img method를 통해 이벤트 발생"""
        self.img_label.setPixmap(QPixmap.fromImage(img))

    def draw_area(self, frame: np.ndarray, box_ltrb: Tuple[int, int, int, int]) -> np.ndarray:
        """인퍼런스 영역과 커서 조작 영역 표시

        Args:
            frame (np.ndarray): 웹캠 이미지
            box_ltrb (tuple): 인퍼런스 영역 좌표, 차례로 박스의 좌, 상, 우, 하 좌표값

        Return:
            인퍼런스 영역과 커서 조작 영역이 표시된 np.ndarray 타입 이미지
        """
        # 인퍼런스 영역 표시
        l, t, r, b = box_ltrb
        frame = cv2.rectangle(frame, (l, t), (r, b), (128, 0, 255), 2)

        # 커서 조작 영역 표시
        ctr_box_size = int(self.__box_size * self.cursor.ctr_area_ratio)
        l_ctr, t_ctr = self.__box_center[0] - (ctr_box_size // 2), t + int(self.__box_size * .05)
        r_ctr, b_ctr = l_ctr + ctr_box_size, t_ctr + ctr_box_size
        frame = cv2.rectangle(frame, (l_ctr, t_ctr), (r_ctr, b_ctr), (255, 0, 128), 2)

        return frame

    # TODO: idx_pos 타입 힌트를 튜플로 수정하고 원소 수 제한 표현
    def draw_idx_point(self, frame: np.ndarray, patch_lt: Tuple[int, int], idx_pos: List[float]) -> np.ndarray:
        """커서(검지) 위치 표시

        Args:
            frame (np.ndarray): 웹캠 이미지
            patch_lt (tuple): 인퍼런스 영역의 좌상 좌표. 전체 이미지에서의 커서 위치 계산에 사용
            idx_pos (list): 인퍼런스 영역 내에서의 커서 좌표

        Return:
            커서 위치가 표시된 np.ndarray 타입 이미지
        """
        l, t = patch_lt
        if idx_pos:
            # if self.use_left_hand:
                # idx_pos[0] = 1. - idx_pos[0]
            idx_x = int(idx_pos[0] * self.__box_size + l)
            idx_y = int(idx_pos[1] * self.__box_size + t)
            frame = cv2.circle(frame, (idx_x, idx_y), 5, (0, 255, 0), -1)

        return frame

    def draw_skeleton(self, frame, kpts, patch_lt, gesture: Optional[None]):
        color = self.skeleton_color[gesture]
        kpts_px = [int(patch_lt[i % 2] + x * self.__box_size) for i, x in enumerate(kpts)]
        for i in range(21):
            frame = cv2.circle(frame, kpts_px[2*i: 2*(i+1)], 5, color, 2)

        keypoints_sequence = (
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
        (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))
        for start, end in keypoints_sequence:
            cv2.line(frame, kpts_px[2*start: 2*(start+1)], kpts_px[2*end: 2*(end+1)], color, 2)

        cv2.circle(frame, kpts_px[16:18], 10, color, 2)

        return frame


    @property
    def box_center(self):
        return None

    @box_center.setter
    def box_center(self, value: Tuple[int, int]):
        """인퍼런스 영역이 이미지를 벗어나지 않도록 영역 중심 설정"""
        print(self.__box_center)
        print(value)
        value = [int(v * self.CAM_LABEL_RATIO) for v in value]
        value[0] = self.CAM_W - min(max(value[0], self.__box_size//2), self.CAM_W - self.__box_size//2)
        value[1] = min(max(value[1], self.__box_size//2), self.CAM_H - self.__box_size//2)
        self.__box_center = value

    @property
    def cursor_ctr_ratio(self):
        return None

    @cursor_ctr_ratio.setter
    def cursor_ctr_ratio(self, val: float):
        """Cursor 객체의 커서 조작 영역 비율 갱신"""
        self.cursor.ctr_area_ratio = val


class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('virtual mouse')

        self.th = CursorThread(self, self.fps, self.cam_img)
        self.save_btn()  # 초기 설정 반영
        self.th.start()

        self.save.clicked.connect(self.save_btn)  # 설정 변경 메소드 연결
        self.cam_img.mousePressEvent = self.set_infer_area  # 인퍼런스 영역 변경 메소드 연결

        # 라벨에 커서 호버시에 도움말을 상태표시줄에 표시
        self.text_delay.enterEvent = self.help_delay
        self.text_delay.leaveEvent = self.help_reset
        self.text_hand.enterEvent = self.help_hand
        self.text_hand.leaveEvent = self.help_reset
        self.text_ctr_area.enterEvent = self.help_ctr_area
        self.text_ctr_area.leaveEvent = self.help_reset
        self.help_bar.showMessage('')

    def save_btn(self):
        """버튼 클릭 이벤트 - 유저 설정 적용"""
        delay = self.delay.value()
        ctr_ratio = self.ctr_area_ratio.value()
        use_left_hand = True if self.left_hand.isChecked() else False

        self.th.delay = delay
        self.th.cursor_ctr_ratio = ctr_ratio
        self.th.use_left_hand = use_left_hand

    def set_infer_area(self, event):
        self.th.box_center = int(event.x()), int(event.y())

    def help_delay(self, _):
        self.help_bar.showMessage('커서 조작 딜레이(초단위)')

    def help_hand(self, _):
        self.help_bar.showMessage('왼손 조작 / 오른손 조작 선택')

    def help_ctr_area(self, _):
        self.help_bar.showMessage('인퍼런스 영역 대비 조작 영역의 비율')

    def help_reset(self, _):
        self.help_bar.showMessage('')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
    myWindow.th.exit()
