'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
import pyautogui
import multiprocessing as mp

class MouseController:
    def __init__(self, precision, speed):
        precision_dict={'high':10, 'low':1000, 'medium':500}
        speed_dict={'fast':0.001, 'slow':10, 'medium':5}
        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]

    def move(self, x, y):
        pyautogui.moveRel(x*self.precision, -1*y*self.precision, duration=self.speed)
    def position(self):
        x, y= pyautogui.position()
        return x,y
def mover(x,y):
    mouse = MouseController('high','fast')
    mouse.move(0,-10)
    print(x,y)

def main():
    print("Main init")
    pool = mp.Pool(1)
    x, y = 1,1
    print(mp.cpu_count())
    pool.apply(mover,args=(x,y))

if __name__ == '__main__':
    main()