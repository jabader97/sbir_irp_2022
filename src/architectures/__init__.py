import torch.nn
from torchvision import models
import timm
from architectures.senet import cse_resnet50, CSEResnetModel_KDHashing

supported_models = {'sake': ['cse_resnet50', 'cse_resnet50_hashing'],
                    'sem_pcyc': ['resnet50', 'se_resnet50', 'vgg']}


def get_model(arch, out_dim, model, hashing_dim=0, freeze_features=False, ems=False):
    if arch not in supported_models[model]:
        raise ValueError("{} not supported for {}".format(arch, model))
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
    elif arch == 'cse_resnet50':
        model_arch = cse_resnet50()
    elif arch == 'cse_resnet50_hashing':
        model_arch = CSEResnetModel_KDHashing(hashing_dim, out_dim, freeze_features=freeze_features, ems=ems)
    else:
        raise ValueError("Architecture {} not supported".format(arch))
    return model_arch
