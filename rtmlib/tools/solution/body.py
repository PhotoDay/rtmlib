'''
Example:

import cv2

from rtmlib import Body, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture('./demo.mp4')

openpose_skeleton = True  # True for openpose-style, False for mmpose-style

body = Body(to_openpose=openpose_skeleton,
                      backend=backend,
                      device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = body(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.43)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)

'''
import numpy as np

class Body:
    MODE = {
        'performance': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',  # noqa
            'pose_input_size': (288, 384),
        }
    }

    def __init__(self,
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):

        from .. import YOLOX, RTMPose

        cfg = self.MODE['performance']

        self.det_model = YOLOX(cfg['det'],
                               model_input_size=cfg['det_input_size'],
                               backend=backend,
                               device=device)

        self.pose_model = RTMPose(cfg['pose'],
                                  model_input_size=cfg['pose_input_size'],
                                  to_openpose=to_openpose,
                                  backend=backend,
                                  device=device)

    def __call__(self, image: np.ndarray):
        bboxes = self.det_model(image)
        keypoints, scores = self.pose_model(image, bboxes=bboxes)
        return keypoints, scores, bboxes
