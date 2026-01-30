import torch
import torch.nn as nn
import torchvision.models as models

def get_model(cfg):
    model_name = cfg['model']['name']
    num_classes = cfg['model']['num_classes']
    pretrained = cfg['model']['pretrained']
    freeze_base = cfg['model']['freeze_base']

    
    model = getattr(models, model_name)(pretrained=pretrained)

    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False

    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )

    return model