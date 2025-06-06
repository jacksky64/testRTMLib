import cv2
import numpy as np

from rtmlib import Wholebody, Body, Hand, draw_skeleton, draw_bbox, YOLOX, RTMPose, RTMO

import time
import os

device = 'cpu'  # cpu, cuda, mps
backend = 'openvino'  # opencv, onnxruntime, openvino

inputFolder = './data/input'
outpurFolder = './data/output'
if not os.path.exists(outpurFolder):
    os.makedirs(outpurFolder)   

if not os.path.exists(inputFolder):
    FileNotFoundError(f"Input folder {inputFolder} does not exist. Please create it and add an image file.")

inputFileNameExt = os.path.join(inputFolder, 'sample4.png')

outputPrefixFileName = os.path.join(outpurFolder, os.path.basename(os.path.splitext(inputFileNameExt)[0]))

if not os.path.exists(inputFileNameExt):
    raise FileNotFoundError(f"Input file {inputFileNameExt} does not exist. Please add an image file to the input folder.") 


img = cv2.imread(inputFileNameExt)

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

# key poinyt description
# https://user-images.githubusercontent.com/100993824/227770977-c8f00355-c43a-467e-8444-d307789cf4b2.png
# 1: nose, 1: left eye, 3: right eye, 4: left ear, 5: right ear,
# 6: left shoulder, 7: right shoulder, 8: left elbow, 9: right elbow,
# 10: left wrist, 11: right wrist, 12: left hip, 13: right hip,
# 14: left knee, 15: right knee, 16: left ankle, 17: right ankle

# model-zoo 
# https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose#-model-zoo-



                

# # hand
# hand = Hand(det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
#             det_input_size=(640, 640),
#             pose='https://github.com/open-mmlab/mmpose/blob/dev-1.x/projects/rtmpose/rtmdet/hand/rtmdet_nano_320-8xb32_hand.py',
#             #'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-hand7_pt-hand7_700e-384x288-71d7b7e9_20230629.zip',
#             pose_input_size=(320, 320),
#             backend=backend,
#             device=device)
# keypoints, scores = hand(img)
# # visualize
# # if you want to use black background instead of original image,
# img_show = np.zeros(img.shape, dtype=np.uint8)
# img_show = img.copy()
# img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)

# outputFileName = inputFileName + 'hand.png'
# cv2.imwrite(outputFileName, img_show)
# cv2.imshow(outputFileName, img_show)
# cv2.waitKey(0)




# wholebody 
wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)

keypoints, scores = wholebody(img)
img_show = img.copy()

img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)

outputFileName = outputPrefixFileName + '_wholebody.png'
cv2.imwrite(outputFileName , img_show)
cv2.imshow(outputFileName, img_show)


#RTMO 

# use detector only for bbx detection
det= 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip'
det_input_size = (640, 640)

det_model = YOLOX(det,
            model_input_size=det_input_size,
            backend=backend,
            device=device)
bboxes = det_model(img)

# use pose model for keypoint detection w/out bbx detection
pose='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip'
pose_input_size = (640, 640)

pose_model = RTMO(pose,
            model_input_size=pose_input_size,
            to_openpose=openpose_skeleton,
            backend=backend,
            device=device)

keypoints, scores = pose_model(img)

img_show = img.copy()
img_show = draw_bbox(img_show, bboxes, color=(0, 255, 0))

img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)
outputFileName = outputPrefixFileName + '_rtmo_key_yolox_box.png'
cv2.imwrite(outputFileName, img_show)
cv2.imshow(outputFileName, img_show)


# det= 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip'
det= 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip'
det_input_size = (640, 640)
pose='https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip'
pose_input_size = (288, 384)

det_model = YOLOX(det,
            model_input_size=det_input_size,
            backend=backend,
            device=device)
pose_model = RTMPose(pose,
            model_input_size=pose_input_size,
            to_openpose=openpose_skeleton,
            backend=backend,
            device=device)

bboxes = det_model(img)
keypoints, scores = pose_model(img, bboxes=bboxes)

img_show = img.copy()
for bbox in bboxes:
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 2)

img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)

outputFileName = outputPrefixFileName + '_yolox_rtmpose.png'
cv2.imwrite(outputFileName, img_show)
cv2.imshow(outputFileName, img_show)



# body
body = Body(to_openpose=openpose_skeleton,
                      mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)
keypoints, scores = body(img)

# visualize

# if you want to use black background instead of original image,
img_show = np.zeros(img.shape, dtype=np.uint8)
img_show = img.copy()

img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)
outputFileName = outputPrefixFileName + '_body.png'    
cv2.imwrite(outputFileName, img_show)
cv2.imshow(outputFileName, img_show)


# body with model specification
body = Body(det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
            det_input_size=(640, 640),
            pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
            pose_input_size=(288, 384),
            backend=backend,
            device=device)
keypoints, scores = body(img)

# visualize
img_show = img.copy()

img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)
outputFileName = outputPrefixFileName + '_body_models_yolox_rtmpose_x.png'
cv2.imwrite(outputFileName, img_show)
cv2.imshow(outputFileName, img_show)
#cv2.waitKey(0)

