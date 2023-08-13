import torch
from super_gradients.training import models

best_model = models.get('yolo_nas_l',
                        num_classes=2,
                        checkpoint_path="./checkpoints/warehouse-model-2/ckpt_best.pth")

device = 0 if torch.cuda.is_available() else "cpu"

input_video_path = "./model-test.mp4"
output_video_path = "./model-detect-3.mp4"
#device=0

best_model.to(device).predict(input_video_path,conf=0.65).save(output_video_path)
