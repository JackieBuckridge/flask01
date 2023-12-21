
import mmcv

from mmaction2.mmaction.apis import inference_recognizer,init_recognizer

# Set up the necessary paths
config_file = 'checkpoints/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
checkpoint_file = './checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
video_file = 'demo.mp4'  # Replace with the actual path to your video file

model = init_recognizer(config_file, checkpoint_file, device="cpu")  # device can be 'cuda:0'
# test a single image
result = inference_recognizer(model, video=video_file)