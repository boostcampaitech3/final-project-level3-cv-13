from typing import Tuple, TypeVar, Union, Optional
from dataclasses import dataclass
from time import time
from collections import namedtuple
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
INDEX_T = mp_hands.HandLandmark.INDEX_FINGER_TIP
MIDDLE_P = mp_hands.HandLandmark.MIDDLE_FINGER_PIP
THUMB_T = mp_hands.HandLandmark.THUMB_TIP

GestureType = str
Coord = Tuple[float, float]


@dataclass
class FingerXY:
    idx_t: Optional[Tuple[float, float]] = None
    mid_p: Optional[Tuple[float, float]] = None
    thu_t: Optional[Tuple[float, float]] = None
    kpts = None

    def __init__(self, results):
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.idx_t = (hand_landmarks.landmark[INDEX_T].x, hand_landmarks.landmark[INDEX_T].y)
            self.mid_p = (hand_landmarks.landmark[MIDDLE_P].x, hand_landmarks.landmark[MIDDLE_P].y)
            self.thu_t = (hand_landmarks.landmark[THUMB_T].x, hand_landmarks.landmark[THUMB_T].y,)
            self.kpts = hand_landmarks.landmark


class GestureRecognition:
    def __init__(self, ctr_area_ratio: float):
        self.pre_gesture = None
        self.ctr_area_ratio = ctr_area_ratio
        self.prev_info = None
        self.t_mouse_down = None
        self.prev_gesture = None

    def __call__(self,
                 results,
                 frame: np.ndarray = None):
        fig_xy = FingerXY(results)
        cursor_pos, valid = self.get_cursor_pos(fig_xy)
        gesture = self.classify_gesture(fig_xy)
        # print(gesture)

        return fig_xy, gesture, cursor_pos, valid

    def classify_gesture(self, fig_xy: FingerXY) -> GestureType:
        if fig_xy.mid_p is not None:
            dist_mid2thu = np.linalg.norm(np.array(fig_xy.mid_p)-np.array(fig_xy.thu_t))
            # print(dist_mid2thu)
            if dist_mid2thu >= .1 and self.prev_gesture == 'mouse_down':
                self.prev_gesture = 'mouse_up'
                self.t_mouse_down = None
                return 'mouse_up'
            elif dist_mid2thu < .1:
                if self.prev_gesture != 'mouse_down':
                    self.prev_gesture = 'mouse_down'
                    self.t_mouse_down = time()
                    return 'mouse_down'
                elif self.t_mouse_down is not None:
                    if time() - self.t_mouse_down >= 1.:
                        return 'drag'
                    else:
                        return 'stop'
        return 'move'

    def get_cursor_pos(self, fig_xy: FingerXY):
        c_ratio = self.ctr_area_ratio
        if fig_xy.idx_t is None:
            return None, False
        cursor_pos = (np.clip(fig_xy.idx_t, c_ratio, 1-c_ratio) - c_ratio) / (1 - 2*c_ratio)

        valid = True

        pips = np.array([(fig_xy.kpts[i].x, fig_xy.kpts[i].y) for i in (18, 14, 10)]).reshape(-1, 2)
        pips_avg = np.average(pips, axis=0)

        if self.prev_info is not None:
            if np.linalg.norm(pips_avg-self.prev_info) < 0.0025:
                valid = False
            print(np.linalg.norm(pips_avg - self.prev_info))
            self.prev_info = pips_avg

        else:
            self.prev_info = pips_avg
        # if len(self.prev_cursor_pos) == 3:
        #     a, b, c = self.prev_cursor_pos
        #     dir_1, dir_2 = b-a, c-b
        #     unit_1, unit_2 = dir_1 / np.linalg.norm(dir_1), dir_2 / np.linalg.norm(dir_2)
        #     if np.dot(unit_1, unit_2) < .75:
        #         valid = False

        return cursor_pos, valid

