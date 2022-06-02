from typing import List
import mediapipe as mp
import numpy as np


class MediapipeHandModule:
    """mediapipe의 hands 모듈을 통한 hand keypoints estimation 수행"""
    def __init__(self,
                 model_complexity: int = 1,
                 min_detection_confidence: float = .5,
                 min_tracking_confidence: float = .5,):
        self.hands = mp.solutions.hands.Hands(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __call__(self, img: np.ndarray) -> List[float]:
        kpts = self.hands.process(img)
        kpts_list = []
        if kpts.multi_hand_landmarks:
            hand_landmarks = kpts.multi_hand_landmarks[0]  # 한 손만 처리
            for i in range(21):
                xy = [hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y]
                kpts_list.extend(xy)

        return kpts_list


class OnnxHandModule:
    def __init__(self, model_path: str):
        raise NotImplementedError
