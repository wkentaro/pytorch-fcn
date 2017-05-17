import torch
import torchvision


def VGG16(pretrained=False):
    model = torchvision.models.vgg16(pretrained=False)
    if not pretrained:
        return model
    model_url = 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth'  # NOQA
    state_dict = torch.utils.model_zoo.load_url(model_url)
    # patch state_dict
    state_dict['classifier.0.weight'] = state_dict.pop('classifier.1.weight')
    state_dict['classifier.0.bias'] = state_dict.pop('classifier.1.bias')
    state_dict['classifier.3.weight'] = state_dict.pop('classifier.4.weight')
    state_dict['classifier.3.bias'] = state_dict.pop('classifier.4.bias')
    model.load_state_dict(state_dict)
    return model
