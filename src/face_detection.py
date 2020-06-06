'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import sys
import os
import time
def roundx(x):
    if x[0]<0:
        x[0]=0
    return x
class Model_face_detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold = 0.4):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.core=None
        self.net = None
        self.model = None
        self.model_structure =model_name+'.xml'
        self.model_weights =model_name+'.bin'
        self.device = device
        self.threshold = threshold

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.net = self.core.load_network(network=self.model,device_name=self.device)
        print("Model loaded")

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        crop_frame= []
        out_frame = image
        self.height, self.width, self.channels = image.shape
        input_image = self.preprocess_input(image)
        infer_time = time.time() 
        self.net.infer({self.input_blob:input_image})
        print("Face infer time is {}".format(time.time()-infer_time))
        self.output_blob = next(iter(self.model.outputs))
        output = self.net.requests[0].outputs[self.output_blob]
        coords, detect = self.preprocess_output(output)
        if detect:
            out_frame = self.draw_outputs(coords, image)
            crop_frame =self.crop(coords,image)
        return out_frame, crop_frame

    def draw_outputs(self, coords, image):
        for coord in coords:
            cv2.rectangle(image,(coord[0],coord[1]),(coord[2],coord[3]),(0,255,0),1)
        return image
    def crop(self,coords, image):
        for coord in coords:
            crop_image= image[coord[1]:coord[3],coord[0]:coord[2]]
        return crop_image        
    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        self.input_blob = next(iter(self.model.inputs))
        shape = self.model.inputs[self.input_blob].shape
        frame = cv2.resize(image, (shape[3],shape[2]))
        frame =frame.transpose((2,0,1))
        frame = frame.reshape(1,*frame.shape)
        return frame

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        flag = False
        arr = outputs.flatten()
        matrix = np.reshape(arr,(-1,7))
        matrix = [item for item in matrix if item[2]> self.threshold]
        if len(matrix)>0:
            flag = True
            *matrix, = map(lambda x: x[3:7],matrix)
            matrix = np.array(matrix)
            matrix[:,0] =matrix[:,0]*self.width
            matrix[:,2] =matrix[:,2]*self.width
            matrix[:,1] =matrix[:,1]*self.height
            matrix[:,3] =matrix[:,3]*self.height
            *matrix, = map(lambda x: list(map(int,x)), matrix)
            *matrix, = map(roundx, matrix)
        return matrix, flag
