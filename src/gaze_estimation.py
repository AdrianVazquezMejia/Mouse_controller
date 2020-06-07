'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2
class Model_gaze:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', _extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.core=None
        self.net = None
        self.model = None
        self.model_structure =model_name+'.xml'
        self.model_weights =model_name+'.bin'
        self.device = device
        self.output_blob = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.net = self.core.load_network(network=self.model,device_name=self.device)
        print("Gaze model loaded")

    def predict(self, right_eye, left_eye, angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        coords = None 
        if len(right_eye)>0 and len(left_eye)>0 and len(angles)>0:
            input_image1,input_image2  = self.preprocess_input(right_eye,left_eye)
            input_dict = {'right_eye_image':input_image1,'left_eye_image':input_image2,'head_pose_angles':[angles] }
            self.net.infer(input_dict)
            self.output_blob = next(iter(self.model.outputs))
            output = self.net.requests[0].outputs
            coords = self.preprocess_output(output)
        return coords

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, right, left):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        #self.input_blob = next(iter(self.model.inputs))
        im_shape = self.model.inputs['right_eye_image'].shape
        frame1 = cv2.resize(right, (im_shape[3],im_shape[2]))
        frame1 =frame1.transpose((2,0,1))
        frame1 = frame1.reshape(1,*frame1.shape)
        frame2 = cv2.resize(left, (im_shape[3],im_shape[2]))
        frame2 = frame2.transpose((2,0,1))
        frame2 = frame2.reshape(1,*frame2.shape)
        return frame1,frame2

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords= outputs[self.output_blob].flatten()
        return coords
