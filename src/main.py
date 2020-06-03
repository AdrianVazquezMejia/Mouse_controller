from  face_detection import Model_face_detection
from  facial_landmarks_detection import Model_landmarks
from  head_pose_estimation import Model_pose
from argparse import ArgumentParser
import cv2
import os
import sys
import logging as log
import time
def build_argparser():
    parser= ArgumentParser()
    parser.add_argument("-f","--face", required=False,default='/home/adrian-estelio/Documents/vision/intel/face-detection-retail-0005/FP32/face-detection-retail-0005')
    parser.add_argument("-l","--landmarks", required=False,default='/home/adrian-estelio/Documents/vision/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009')
    parser.add_argument("-p","--head", required=False,default='/home/adrian-estelio/Documents/vision/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001')
    parser.add_argument("-i","--input", required=False,default='/home/adrian-estelio/Documents/vision/Mouse_controller/resources/image.jpg')
    return parser

def infer_on_stream(args):
    face_model = Model_face_detection(args.face)
    face_model.load_model()
    landmarks_model = Model_landmarks(args.landmarks)
    landmarks_model.load_model()
    head_model = Model_pose(args.head)
    head_model.load_model()
    single_image = False

    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image = True
        input_stream = args.input
    else:
        input_stream = args.input
        if not os.path.isfile(args.input):
            log.error("Specified input file doesn't exist")
            sys.exit(1)

    cap = cv2.VideoCapture(input_stream)
    if input_stream :
        cap.open(args.input)
    if not cap.isOpened():
        log.error("Unable to open source")
    while cap.isOpened():
        flag,image = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(1)
        star= time.time()
        frame, face = face_model.predict(image)
        print("time in land is {} ms ".format(time.time()-star))
        
        land = landmarks_model.predict(face)
        head_model.predict(face)
        cv2.imshow('frame',frame)        
        if single_image:
            cv2.imwrite('frame.jpg',frame)
            break
        if key_pressed == 27:
            break

def main():
    print("Main init")
    args =build_argparser().parse_args()
    infer_on_stream(args)
   
if __name__ == '__main__':
    main()