import mmcv

from mmaction2.mmaction.apis import inference_recognizer ,init_recognizer

config_path = 'check_poinst2/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
checkpoint_path = 'check_poinst2/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth' # can be a local path
img_path = 'demo.mp4'   # you can specify your own picture path


model=init_recognizer(config_path,checkpoint_path,device='cpu')

video=mmcv.VideoReader(img_path)


frame_rate=video.fps

total_frames=len(video)

result=inference_recognizer(model,img_path)

print(result)

# Output the results
for i, (label, score) in enumerate(zip(result['label'], result['score'])):
    print(f'Frame {i+1}: Label - {label}, Score - {score}')

# Load the label map
with open("mmaction2/tools/data/kinetics/label_map_k400.txt", "r") as f:
    label_map = [line.strip() for line in f]

# Map labels to class names
class_names = [label_map[label] for label in result['label']]

print("Class Names:", class_names)