import mmcv

from mmaction2.mmaction.apis import inference_recognizer, init_recognizer

config_path = 'mmaction2/configs/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py'
checkpoint_path = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth' # can be a local path
img_path = 'demo.mp4'   # you can specify your own picture path


# Initialize the recognizer
model = init_recognizer(config_path, checkpoint_path, device='cpu')

# Read the image
img = mmcv.imread(img_path)

# Perform inference
result = inference_recognizer(model, img_path)
print(type(result))
print(dir(result))


for i in result.items():
    break
    print(i)

label="mmaction2/tools/data/kinetics/label_map_k400.txt"
print(label)