import torch.nn
from torchvision import models
import timm

supported_models = {'sake': ['cse_resnet50'],
                    'sem_pcyc': ['resnet50', 'se_resnet50', 'vgg']}


def get_model(params_model):
    if params_model['image_arch'] not in supported_models[params_model['model']] or \
            params_model['sketch_arch'] not in supported_models[params_model['model']]:
        raise ValueError("{} not supported for SAKE")
    image_model = get(params_model['image_arch'], params_model['image_dim'])
    sketch_model = get(params_model['sketch_arch'], params_model['sketch_dim'])
    return image_model, sketch_model


def get(arch, out_dim):
    if arch == 'resnet50':
        model_arch = timm.create_model('resnet50', pretrained=True)
        model_arch.fc = torch.nn.Linear(in_features=2048, out_features=out_dim)
    elif arch == 'se_resnet50':
        model_arch = timm.create_model('seresnet50', pretrained=True)
        model_arch.fc = torch.nn.Linear(in_features=2048, out_features=out_dim)
    elif arch == 'vgg':
        model_arch = models.vgg16(pretrained=True)
        model_arch.classifier = torch.nn.Sequential(torch.nn.Sequential(*list(model_arch.classifier.children())[:-1]),
                                                    torch.nn.Linear(in_features=4096, out_features=out_dim))
    else:
        raise ValueError("Architecture {} not supported".format(arch))
    return model_arch
