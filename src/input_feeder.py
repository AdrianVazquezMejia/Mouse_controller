'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type=input_type
        if input_type in ('video', 'image'):
            self.input_file=input_file
        if input_type == 'CAM':
            self.input_file = 0
        self.opened = False
        self.cap = None
        self.width = None
        self.height = None
        self.fps = None
    
    def load_data(self):
        self.cap=cv2.VideoCapture(self.input_file)
        self.width = int(self.cap.get(3))
        self.height = int(self.cap.get(4))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def open(self):
        if self.input_file:
            self.cap.open(self.input_file) 
        self.opened = self.cap.isOpened()
    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
                    
        self.opened, frame=self.cap.read()
        return frame


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

