# Command to convert YOLOv8 to TensorRT engine 

BATCH_SIZE=5
/opt/vision_dependencies/tensorrt/TensorRT-${TENSORRT_VERSION}/targets/x86_64-linux-gnu/bin/trtexec \
    --onnx=/home/docker/modules/pose-inference/models/rtmpose${BATCH_SIZE}.onnx \
    --shapes=images:${BATCH_SIZE}x3x640x640 \
    --saveEngine=/home/docker/modules/pose_inference/models/rtmpose${BATCH_SIZE}.engine

# Command to convert dynamic onnx model to fixed shape
python3 -m onnxruntime.tools.make_dynamic_shape_fixed   --input_name input   --input_shape 5,3,256,192   /home/docker/modules/pose-inference/models/rtmpose.onnx   /home/docker/modules/pose-inference/models/rtmpose5_fixed.onnx
