from . import resnet

models = {
    'resnet50': resnet.ResNet50,
    'resnet101': resnet.ResNet101,
    'resnet152': resnet.ResNet152,
}
    