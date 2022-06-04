import time
from typing import List, Tuple, Optional

import numpy as np
import pyautogui


IDX2GESTURE = {
    0: 'mouse_up',
    1: 'mouse_down',
    2: 'right_click',
    3: None,
    4: None,
}


class Cursor:
    """커서 조작"""
    def __init__(self, ctr_area_ratio: float):
        self.ctr_area_ratio = ctr_area_ratio
        self.scr_wh = pyautogui.size()
        self.prev_gesture = None
        self.t_down = None

    def get_cursor_pos(self, kpts: List[float]) -> Tuple[Tuple[float, float], List]:
        """커서 좌표 계산: 검지 끝 키포인트 좌표 계산"""
        idx_t = kpts[16:18]
        bound = (1. - self.ctr_area_ratio) / 2.
        cursor_pos_x = (np.clip(1. - idx_t[0], bound, 1. - bound) - bound) / (1. - bound * 2)
        cursor_pos_y = np.clip(idx_t[1], 0., 1. - bound * 2.) / (1. - bound * 2)

        return (cursor_pos_x, cursor_pos_y), idx_t

    def get_gesture(self, gesture: str) -> Optional[str]:
        """이전 동작을 고려하여 수행할 동작 반환"""
        if gesture in (None, 'right_click'):
            return gesture

        if gesture == 'mouse_up' and self.prev_gesture in ('mouse_down', 'drag', 'wait'):
            self.t_down = time.time()
            return 'mouse_up'
        elif gesture == 'mouse_down':
            if self.prev_gesture not in ['mouse_down', 'drag', 'wait']:
                self.t_down = time.time()
                return 'mouse_down'
            elif time.time() - self.t_down >= 1.:  # 버튼이 눌린 상태로 1초 유지 -> 드래그
                return 'drag'
            else:
                return 'wait'

        return 'move'

    def control(self, pos: Tuple[int, int], gesture: Optional[str]):
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

    def __call__(self, kpts: List[float], gesture_idx: Optional[int]):
        if gesture_idx is not None:
            cursor_pos, idx_pos = self.get_cursor_pos(kpts)
            cursor_pos_pixel = int(cursor_pos[0] * self.scr_wh[0]), int(cursor_pos[1] * self.scr_wh[1])

            gesture = self.get_gesture(IDX2GESTURE[gesture_idx])
            self.control(cursor_pos_pixel, gesture)
            self.prev_gesture = gesture

            return idx_pos if gesture else None









