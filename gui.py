import sys
import argparse
from time import sleep, time

import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import *

from inference import MediapipeHandModule, OnnxHandModule
from hand_gesture import GestureRecognition
from cursor_control import Cursor


parser = argparse.ArgumentParser()
parser.add_argument(
    '--kpts_model_path', type=str, default=None,
    help='키포인트 모델 경로, None인 경우 mediapipe 사용'
)
parser.add_argument(
    '--gesture_model_path', type=str, default='./prototype/hand_gesture_classification/simple_model.onnx',
    help='손동작 분류 모델 경로'
)
args = parser.parse_args()


form_class = uic.loadUiType("gui_test.ui")[0]


class CursorThread(QThread):
    change_pixmap = pyqtSignal(QImage)

    # def __init__(self, p, time_per_step_label, img_label, delay=.1, ctr_ratio=.8):
    def __init__(self, p, time_per_step_label, img_label):
        super().__init__(p)
        self.delay = None
        self.use_left_hand = None
        self.time_per_step_label = time_per_step_label
        self.img_label = img_label
        self.change_pixmap.connect(self.update_img)

        self.W, self.H = 640, 480
        self.show_W, self.show_H = 320, 240
        self.ratio = self.W / self.show_W

        self.__box_center = (200, 320)
        self.__box_size = 244

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
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FPS, 30)

        while True:
            t_1 = time()
            ret, frame = capture.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            l, t = self.__box_center[0] - self.__box_size // 2, self.__box_center[1] - self.__box_size // 2
            r, b = l + self.__box_size, t + self.__box_size

            patch = np.ascontiguousarray(frame[t: b, l: r])
            if self.use_left_hand:
                patch = cv2.flip(patch, 1)
            patch.flags.writeable = False

            kpts = self.kpts_model(patch)
            if kpts:
                hand_pose = self.gesture_recognition(kpts)
                idx_pos = self.cursor(kpts, hand_pose)
                self.draw_idx_point(frame, (l, t), idx_pos)

            frame = self.draw_area(frame, (l, t, r, b))

            cv2.circle(frame, self.__box_center, 5, (255, 0, 0), -1)
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            bytes_per_line = w * c
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).scaled(self.show_W, self.show_H)
            self.change_pixmap.emit(q_img)
            self.time_per_step_label.setText(f'{time()-t_1:.06f} s')
            sleep(self.delay)

    def update_img(self, img):
        self.img_label.setPixmap(QPixmap.fromImage(img))

    def draw_area(self, frame, box_ltrb):
        l, t, r, b = box_ltrb
        frame = cv2.rectangle(frame, (l, t), (r, b), (128, 0, 255), 2)

        ctr_box_size = int(self.__box_size * self.cursor.ctr_area_ratio)
        l_ctr = self.__box_center[0] - (ctr_box_size // 2)
        t_ctr = t
        r_ctr = l_ctr + ctr_box_size
        b_ctr = t_ctr + ctr_box_size
        frame = cv2.rectangle(frame, (l_ctr, t_ctr), (r_ctr, b_ctr), (255, 0, 128), 2)

        return frame

    def draw_idx_point(self, frame, patch_lt, idx_pos):
        l, t = patch_lt
        if idx_pos:
            if self.use_left_hand:
                idx_pos[0] = 1. - idx_pos[0]
            idx_x = int(idx_pos[0] * self.__box_size + l)
            idx_y = int(idx_pos[1] * self.__box_size + t)
            frame = cv2.circle(frame, (idx_x, idx_y), 5, (0, 255, 0), -1)

        return frame

    @property
    def box_center(self):
        return None

    @box_center.setter
    def box_center(self, value):
        value = [int(v * self.ratio) for v in value]
        value[0] = self.W - min(max(value[0], self.__box_size//2), self.W - self.__box_size//2)
        value[1] = min(max(value[1], self.__box_size//2), self.H - self.__box_size//2)
        self.__box_center = value

    @property
    def cursor_ctr_ratio(self):
        return None

    @cursor_ctr_ratio.setter
    def cursor_ctr_ratio(self, val):
        self.cursor.ctr_area_ratio = val


class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # self.th = CursorThread(self, self.time_per_step, self.cam_img, self.delay.value(), self.ctr_area_ratio.value())
        self.th = CursorThread(self, self.time_per_step, self.cam_img)
        self.save_btn()
        self.th.start()

        self.save.clicked.connect(self.save_btn)
        self.cam_img.mousePressEvent = self.set_infer_area

        self.text_delay.enterEvent = self.help_delay
        self.text_delay.leaveEvent = self.help_reset
        self.text_hand.enterEvent = self.help_hand
        self.text_hand.leaveEvent = self.help_reset
        self.help_bar.showMessage('')

    def save_btn(self):
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

    def help_reset(self, _):
        self.help_bar.showMessage('')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
    myWindow.th.exit()
