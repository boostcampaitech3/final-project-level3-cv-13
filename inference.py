from typing import List
import numpy as np
import mediapipe as mp
import onnxruntime

from prototype.functions import preprocessing, keypoints_from_heatmaps


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
    CENTER: np.ndarray = np.array([112., 112.], dtype=np.float32)[np.newaxis]
    # SCALE: np.ndarray = np.array([0.896, 0.896], dtype=np.float32)[np.newaxis]
    SCALE: np.ndarray = np.array([1., 1.], dtype=np.float32)[np.newaxis]

    def __init__(self, model_path: str):
        self.hands = onnxruntime.InferenceSession(model_path)

    def __call__(self, img: np.ndarray) -> List[float]:
        img_prep = preprocessing(img)
        ort_inputs = {self.hands.get_inputs()[0].name: img_prep}
        ort_outs = self.hands.run(None, ort_inputs)
        kpts_with_score = keypoints_from_heatmaps(heatmaps=ort_outs[0], center=self.CENTER, scale=self.SCALE)

        mean_score = kpts_with_score[1].mean()
        kpts = kpts_with_score[0]
        kpts_norm = kpts / np.array(img.shape[1::-1])
        return kpts_norm.flatten().tolist() if mean_score >= .5 else None


if __name__ == '__main__':
    fake_input = np.random.normal(size=(224, 224, 3))
    onnx_hands = OnnxHandModule('m3_cv7a_224x224_220605.onnx')
    print(onnx_hands(fake_input))
