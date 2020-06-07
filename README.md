# Computer Pointer Controller

This project uses OpenVINO toolkit from Intel to deploy an app that can controll the mouse pointer of a computer using  face gestures. 

## Project Set Up and Installation

To use this project:
* Clone this repository `git clone https://github.com/AdrianVazquezMejia/Mouse_controller.git`
* Download and install OpenVINO from (here)[https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html]
* Download the following models:
-  face-detection-retail-0005
-  landmarks-regression-retail-0009
-  head-pose-estimation-adas-0001
-  gaze-estimation-adas-0002
*  Open the file `mouse.sh` in the 3rd line input the paths accodingly. You can use the default to get help about it.
* To run the project use the command `source mouse.sh`.

## Demo
There is a video input demo which the  mouse control is calibrated which is `local4.mp4`.
Select this as input to verify everything is working properly. Fill the gaps in `demo.sh` and then run `source demo.sh`
Now you should watch a windows with my face making some weird movements and your mouse pointer moving accordingly.
## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

* `main.py` handles the appm meanwhile the includes with names relates to models have the objetcs defitions to handle them.
`Model_controller.py` and `input_feeder.py` have a similar function.

* Inputs argument uses and help can be get using `python3 main.py --help`:
```usage: main.py [-h] [--face FACE] [--landmarks LANDMARKS] [--head HEAD]
               [--gaze GAZE] [--input INPUT] [--visual_o VISUAL_O]
               [--device DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  --face FACE           Face detecion model path
  --landmarks LANDMARKS
                        landmarks detection model path
  --head HEAD           head pose estimation model path
  --gaze GAZE           Gaze estimation model path
  --input INPUT         Input: image or video path or webcam (CAM)
  --visual_o VISUAL_O   Flag to display face: True or False```

## Benchmarks
* Loading time: The FP32 models takes less time to load. And CPU is faster in loading than other hardware.
* Inference time: CPU has the best performace with FP32 precision. The larger time for inferece is in `Face detection`, but it is faster in FP32_INT8 precision.
* Pre and post processing does not take any relevant amount of time relative to the infere time.
## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.
* The result are aceptable, the user can controll the mouse pointer in almost real time and with an aceptable accuracy.
* This models runs very fast and are lightweight. IR representations is very optimized.
* FP32 is faster in CPU because can be processed in a 64 bit architecture without that admit FP operations. Other precision does not represent a huge improvement. In addition, it has the best precision of the model.
# License 
MIT License

Copyright (c) [2020] [Adrian Vazquez]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
