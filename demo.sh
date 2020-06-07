source .env/bin/activate
source /opt/intel/openvino/bin/setupvars.sh
python3 src/main.py --input bin/demo.mp4 --face <models_path>/face-detection-retail-0005/FP32-INT8/face-detection-retail-0005 --landmarks <models_path>/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 --head <models_path>/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 --gaze <models_path>/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 --visual_o True --device CPU
