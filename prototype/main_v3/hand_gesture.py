from typing import List, Optional

import numpy as np
import onnxruntime as ort


class GestureRecognition:
    def __init__(self, model_path: str, score_threshold: float = .6):
        self.model = ort.InferenceSession(model_path)
        self.score_threshold = score_threshold

    def preprocess(self, kpts: List[float]) -> np.ndarray:
        """키포인트 좌표 전처리 - min-max 정규화"""
        kpts = np.array(kpts, dtype=np.float32).reshape(-1, 2)
        kpts = (kpts - kpts.min(axis=0)) / (kpts.max(axis=0) - kpts.min(axis=0))
        return kpts.flatten()[None]

    def __call__(self, kpts: List[float]) -> Optional[int]:
        """손의 키포인트를 통해 손동작 분류"""
        kpts_prep = self.preprocess(kpts)
        pred = self.model.run(['output'], {'input': kpts_prep})[0][0]

        pred_cls = pred.argmax()
        if pred[pred_cls] >= self.score_threshold:
            return pred_cls
        else:
            return None
