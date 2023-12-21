from mmaction2.mmaction.apis import inference_recognizer ,init_recognizer
from mmengine import Config
from operator import itemgetter

config = 'mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
config = Config.fromfile(config)

checkpoint_path="checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth"
video = 'demo.mp4'   # you can specify your own picture path

label="mmaction2/tools/data/kinetics/label_map_k400.txt"


model=init_recognizer(config,checkpoint_path,device='cpu')


results = inference_recognizer(model, video)

pred_scores = results.pred_score.tolist()
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
top5_label = score_sorted[:5]

labels = open(label).readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in top5_label]

print(results)