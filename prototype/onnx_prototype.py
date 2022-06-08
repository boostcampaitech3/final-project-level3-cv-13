import pyautogui  # 마우스 조작
import cv2
import onnxruntime
import numpy as np

from functions import preprocessing, keypoints_from_heatmaps


sc_w, sc_h = pyautogui.size()  # 모니터 해상도: 가로, 세로
cam_w, cam_h = 640, 480  # 캠에서 읽어올 영상의 크기

infer_x_min, infer_x_max = 50, 274  # 캠 영상에서 인퍼런스할 영역
infer_y_min, infer_y_max = 200, 424 ## 지정한 영역만 잘라서 인퍼런스

c = np.array([112., 112.], dtype=np.float32)
c = np.expand_dims(c, axis=0)
s = np.array([0.896, 0.896], dtype=np.float32)
s = np.expand_dims(s, axis=0)

infer_w = infer_x_max - infer_x_min
infer_h = infer_y_max - infer_y_min


ort_session = onnxruntime.InferenceSession("./tmp3.onnx")

# 캠 설정
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)


if capture.isOpened():

    ret, frame = capture.read()
    while ret and cv2.waitKey(5) < 0:  # 5 ms 마다 캠 읽어오기 & 임의의 키를 누르면 while문 탈출
        ret, frame = capture.read()  # 이미지 읽기 / ret: 성공하면 True, frame: 이미지

        infer_patch = frame[infer_y_min: infer_y_max, infer_x_min: infer_x_max]
        infer_patch = cv2.cvtColor(infer_patch, cv2.COLOR_BGR2RGB)
        infer_patch.flags.writeable = False

        ##########################################################################
        
        infer_patch = preprocessing(infer_patch)
        ort_inputs = {ort_session.get_inputs()[0].name: infer_patch}
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_result = keypoints_from_heatmaps(heatmaps=ort_outs[0], center=c, scale=s)
        
        ##########################################################################

        # inference 영역 표시
        cv2.rectangle(frame, (infer_x_min, infer_y_min), (infer_x_max, infer_y_max), (0, 0, 128), 3)
        
        # 검지손가락의 끝마디의 threshold
        if onnx_result[1][0][8][0] > 0.5:

            m_x, m_y = int((1-onnx_result[0][0][8][0]/224) * sc_w), int(onnx_result[0][0][8][1]/224 * sc_h)  # 전체 화면에 대한 좌표 계산
            f_x, f_y = int(infer_x_min + onnx_result[0][0][8][0]), int(infer_y_min + onnx_result[0][0][8][1])  # 검지 좌표

            cv2.circle(frame, (f_x, f_y), 5, (255, 0, 128), -1)  # 캠 영상에 위치 표시
            pyautogui.moveTo(m_x, m_y, _pause=False)  # 마우스 커서 이동
        
        cv2.imshow("VideoFrame", frame)  # 이미지 출력

    capture.release()
    cv2.destroyAllWindows()
