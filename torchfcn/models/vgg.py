import torch
import torchvision


def VGG16(pretrained=False):
    model = torchvision.models.vgg16(pretrained=False)
    if not pretrained:
        return model
    model_file = '/home/wkentaro/data/models/torch/vgg16-from-caffe.pth'
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model
