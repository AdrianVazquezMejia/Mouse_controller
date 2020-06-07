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
moveto=['arriba','abajo', 'izquierda', 'derecha']
def build_argparser():
    parser= ArgumentParser()
    parser.add_argument("-f","--face", required=False,default='/home/adrian-estelio/Documents/vision/intel/face-detection-retail-0005/FP32-INT8/face-detection-retail-0005')
    parser.add_argument("-l","--landmarks", required=False,default='/home/adrian-estelio/Documents/vision/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009')
    parser.add_argument("-p","--head", required=False,default='/home/adrian-estelio/Documents/vision/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001')
    parser.add_argument("-g","--gaze", required=False,default='/home/adrian-estelio/Documents/vision/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002')
    parser.add_argument("-i","--inp", required=False,default='CAM')#/home/adrian-estelio/Documents/vision/Mouse_controller/resources/image.jpg')
    return parser
def move(coor):
    mouse= MouseController('high','fast')
    print("moving")
    if coor[0]<-0.33 and coor[1]>-0.05 and coor[1]<0.05:
        print(moveto[3])
        mouse.move(1,0)
    elif coor[0]>0.33 and coor[1]<0:
        print(moveto[2])
        mouse.move(-1,0)
    elif coor[1]>0.11 and coor[0]>-0.17:
        print(moveto[0])
        mouse.move(0,1)
    elif coor[0]>-0.05  and coor[1]<-0.13:
        print(moveto[1])
        mouse.move(0,-1)
def infer_on_stream(args):
    model_time = time.time()
    face_model = Model_face_detection(args.face)
    face_model.load_model()
    landmarks_model = Model_landmarks(args.landmarks)
    landmarks_model.load_model()
    head_model = Model_pose(args.head)
    head_model.load_model()
    gaze_model = Model_gaze(args.gaze)
    gaze_model.load_model()
    print("Model loading time is {}".format(time.time()-model_time))
    
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
    out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (width,height),True)
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
            start= time.time()    
            _,r,l = landmarks_model.predict(face)
            print("Infer time landmarks is {} s ".format(time.time()-start))
            start= time.time()
            angles= head_model.predict(face)
            print("Infer time head pose is {} s".format(time.time()-start))
            start= time.time()
            vector = gaze_model.predict(r,l,angles)
            print("Infer time gaze is {} s".format(time.time()-start))
            start= time.time()
            move(vector)
            print("Move time is {} s ".format(time.time()-start))
        out.write(frame)
        cv2.imshow('frame',frame)               
        if feeder.input_type == 'image':
            cv2.imwrite('r.jpg',r)
            cv2.imwrite('l.jpg',l)
            cv2.imwrite('frame.jpg',frame)
            break
        if key_pressed == 27:
            break
    out.release()
    feeder.close
def main():
    print("Main init")
    args =build_argparser().parse_args()
    infer_on_stream(args)
   
if __name__ == '__main__':
    main()