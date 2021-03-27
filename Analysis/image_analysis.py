import torch
from PIL import Image
from torchvision import transforms
import torchvision
import numpy as np

# model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
# model.eval()
#
# # filename = "/Users/vashisth/Documents/11-777/group-tars-alfred/tars/alfred/data/json_2.1.0/train/look_at_obj_in_light-AlarmClock-None-DeskLamp-301/trial_T20190907_174127_043461/high_res_images/000000001.png"
# # filename = "/Users/vashisth/Documents/11-777/group-tars-alfred/tars/alfred/data/json_2.1.0/train/look_at_obj_in_light-AlarmClock-None-DeskLamp-301/trial_T20190907_174127_043461/instance_masks/000000001.png"
# filename = "/Users/vashisth/Documents/11-777/group-tars-alfred/Analysis/Dogs.png"
# input_image = Image.open(filename).convert('RGB')
# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
#
# with torch.no_grad():
#     fulloutput = model(input_batch)
#     output = fulloutput['out'][0]
# output_predictions = output.argmax(0)
#
#
# # create a color pallette, selecting a color for each class
# palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
# colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
# colors = (colors % 255).numpy().astype("uint8")
#
# # plot the semantic segmentation predictions of 21 classes in each color
# r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
# r.putpalette(colors)
#
# import matplotlib.pyplot as plt
# plt.imshow(r)
# plt.show()

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

filename = "/Users/vashisth/Documents/11-777/group-tars-alfred/tars/alfred/data/json_2.1.0/train/look_at_obj_in_light-AlarmClock-None-DeskLamp-301/trial_T20190907_174127_043461/high_res_images/000000001.png"
input_image = Image.open(filename).convert('RGB')
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
predictions = model(input_batch)

# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)

print(predictions)

