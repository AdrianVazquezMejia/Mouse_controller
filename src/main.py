from  face_detection import Model_face_detection
from  facial_landmarks_detection import Model_landmarks
from  head_pose_estimation import Model_pose
from  gaze_estimation import Model_gaze
from argparse import ArgumentParser
from mouse_controller import MouseController
from input_feeder import InputFeeder
import cv2
import os
import sys
import logging as log
import time
import multiprocessing as mp
moveto=['up','down', 'left', 'right']
def build_argparser():
    parser= ArgumentParser()
    parser.add_argument("--face", required=False,help= "Face detecion model path ",default='/home/adrian-estelio/Documents/vision/intel/face-detection-retail-0005/FP32-INT8/face-detection-retail-0005')
    parser.add_argument("--landmarks", required=False,help= "landmarks detection model path ", default='/home/adrian-estelio/Documents/vision/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009')
    parser.add_argument("--head", required=False,help= "head pose estimation model path ",default='/home/adrian-estelio/Documents/vision/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001')
    parser.add_argument("--gaze", required=False,help= "Gaze estimation model path ",default='/home/adrian-estelio/Documents/vision/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002')
    parser.add_argument("--input", required=False,help="Input: image or  video path or webcam (CAM) ", default='CAM')#/home/adrian-estelio/Documents/vision/Mouse_controller/resources/image.jpg')
    parser.add_argument("--visual_o",required=False,help="Flag to display face: True or False", default="True")
    parser.add_argument("--device",required=False,help="Device to run the inference", default="CPU")
    return parser
def move(coor):
    mouse= MouseController('high','fast')
    if coor[0]<-0.33 and coor[1]>-0.05 and coor[1]<0.05:
        log.info("Moving to %s",moveto[3])
        mouse.move(1,0)
    elif coor[0]>0.33 and coor[1]<0:
        log.info("Moving to %s",moveto[2])
        mouse.move(-1,0)
    elif coor[1]>0.11 and coor[0]>-0.17:
        log.info("Moving to %s",moveto[0])
        mouse.move(0,1)
    elif coor[0]>-0.05  and coor[1]<-0.13:
        log.info("Moving to %s",moveto[1])
        mouse.move(0,-1)
def infer_on_stream(args):

    face_model = Model_face_detection(args.face,device=args.device)
    face_model.load_model()
    landmarks_model = Model_landmarks(args.landmarks,device=args.device)
    landmarks_model.load_model()
    head_model = Model_pose(args.head,device=args.device)
    head_model.load_model()
    gaze_model = Model_gaze(args.gaze,device=args.device)
    gaze_model.load_model()
    
    if args.input == 'CAM':
        feeder= InputFeeder('CAM')
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        feeder= InputFeeder('image',args.input)
    else:
        feeder= InputFeeder('video',args.input)
        if not os.path.isfile(args.input):
            log.error("Specified input file doesn't exist")
            sys.exit(1)
    feeder.load_data()
    width = feeder.width
    height = feeder.height
    fps = feeder.fps
    out = cv2.VideoWriter('output/out.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (width,height),True)
    feeder.open()
    if not feeder.opened:
        log.error("Unable to open source")
    while feeder.opened:
        image = feeder.next_batch()
        if not feeder.opened:
            break
        key_pressed = cv2.waitKey(1)
        frame, face = face_model.predict(image)
        if len(face)>0:
            _,r,l = landmarks_model.predict(face)
            angles= head_model.predict(face)
            vector = gaze_model.predict(r,l,angles)
            move(vector)
        out.write(frame)
        if args.visual_o == 'True':
            cv2.imshow('frame',frame)               
        if feeder.input_type == 'image':
            cv2.imwrite('output/r.jpg',r)
            cv2.imwrite('output/l.jpg',l)
            cv2.imwrite('output/frame.jpg',frame)
            break
        if key_pressed == 27:
            break
    out.release()
    feeder.close
def main():
    log.basicConfig(level=log.INFO)
    log.info("Aplication started")
    args =build_argparser().parse_args()
    infer_on_stream(args)
   
if __name__ == '__main__':
    main()