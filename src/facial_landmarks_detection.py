from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import sys
import os
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class Model_landmarks:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold = 0.5):
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
        '''crop
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.net = self.core.load_network(network=self.model,device_name=self.device)
        print("Landmarks model loaded")

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.height, self.width, self.channels = image.shape
        print(self.height,self.width)
        input_image = self.preprocess_input(image)
        self.net.infer({self.input_blob:input_image})
        self.output_blob = next(iter(self.model.outputs))
        output = self.net.requests[0].outputs[self.output_blob]
        coords = self.preprocess_output(output)
        out_frame = None
        reye = None 
        leye = None
        if len(coords)>0:
            reye,leye = self.crop_eyes(coords, image)
            out_frame = self.draw_outputs(coords, image)            
        return out_frame, reye, leye
    def crop_eyes(self, coord, image):
        delta = 30
        left = image[coord[3]-delta:coord[3]+delta,coord[2]-delta:coord[2]+delta]
        right = image[coord[1]-delta:coord[1]+delta,coord[0]-delta:coord[0]+delta]     
        cv2.imwrite('r.jpg',right)
        cv2.imwrite('l.jpg',left)
        return right, left
    def draw_outputs(self, coords, image):
        #for coord in coords:
        cv2.circle(image,(coords[0],coords[1]),2,(0,0,255),2)
        cv2.circle(image,(coords[2],coords[3]),2,(0,0,255),2)
        cv2.circle(image,(coords[4],coords[5]),2,(0,0,255),2)
        cv2.circle(image,(coords[6],coords[7]),2,(0,0,255),2)
        cv2.circle(image,(coords[8],coords[9]),2,(0,0,255),2)
        return image
    def check_model(self):
        raise NotImplementedError

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
        print("Face preprocessed successfully")
        return frame

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        arr = outputs.flatten()
        matrix = [arr[i]*self.height if i%2 else arr[i]*self.width for i,_ in enumerate(arr)]
        *matrix, = map(int,matrix)
        print(matrix)
        return matrix
        
