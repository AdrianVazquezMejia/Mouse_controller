'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2
class Model_pose:
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
        self.height = None
        self.width = None 
        self.channels = None
        self.input_blob = None       

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.net = self.core.load_network(network=self.model,device_name=self.device)
        print("Head pose  model loaded")

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.height, self.width, self.channels = image.shape
        input_image = self.preprocess_input(image)
        self.net.infer({self.input_blob:input_image})
        self.output_blob = next(iter(self.model.outputs))
        output = self.net.requests[0].outputs
        coords = self.preprocess_output(output)
        return coords

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
        return frame

    @staticmethod
    def preprocess_output(outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        
        #shape = self.model.outputs[self.output_blob].shape
        pitch = outputs['angle_p_fc'].flatten()
        roll = outputs['angle_r_fc'].flatten()
        yaw = outputs['angle_y_fc'].flatten()
        coords =[yaw, pitch, roll]
        #print(coords)
        return coords
