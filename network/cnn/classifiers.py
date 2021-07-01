import torch
import torchvision


def create_custom_densenet161(class_nb: int,
                              pretrained: bool = False,
                              checkpointing: bool = True) -> torchvision.models.DenseNet:
    """
    Create a DenseNet-161 and modify its last layer to fit with the class number
    Args:
        class_nb: Amount of class
        pretrained: If the network needs ot be pretrained on ImageNet
        checkpointing: If true, enable gradient checkpointing

    Returns:
        A fully initialised Densenet-161
    """
    learner = torchvision.models.densenet161(pretrained=pretrained, memory_efficient=checkpointing)
    num_features = learner.classifier.in_features
    learner.classifier = torch.nn.Linear(num_features, class_nb)

    return learner


def create_custom_resnet50(class_nb: int,
                           pretrained: bool = False) -> torchvision.models.ResNet:
    """
    Create a ResNet-50 and modify its last layer to fit with the class number
    Args:
        class_nb: Amount of class
        pretrained:If the network needs ot be pretrained on ImageNet

    Returns:
        A fully initialised ResNet-50
    """
    learner = torchvision.models.resnet50(pretrained=pretrained)
    num_features = learner.fc.in_features
    learner.fc = torch.nn.Linear(num_features, class_nb)

    return learner


def create_custom_resnet152(class_nb: int,
                            pretrained: bool = False) -> torchvision.models.ResNet:
    """
       Create a ResNet-151 and modify its last layer to fit with the class number
       Args:
           class_nb: Amount of class
           pretrained:If the network needs ot be pretrained on ImageNet

       Returns:
           A fully initialised ResNet-151
   """
    learner = torchvision.models.resnet152(pretrained=pretrained)
    num_features = learner.fc.in_features
    learner.fc = torch.nn.Linear(num_features, class_nb)

    return learner


def create_classifier(name: str,
                      class_nb: int,
                      pretrained: bool = False,
                      checkpointing: bool = True):
    """
    Wrapper function to ease classifier creation
    Args:
        name: Name of the classifier
        class_nb: Amount of class
        pretrained: If the network needs to be pretrained on ImageNet
        checkpointing: If true, enables gradient checkpointing

    Returns:
        A fully initialised classifier
    """
    if name == 'densenet161':
        return create_custom_densenet161(class_nb, pretrained, checkpointing)
    elif name == 'resnet50':
        return create_custom_resnet50(class_nb, pretrained)
    elif name == 'resnet152':
        return create_custom_resnet152(class_nb, pretrained)
    raise NotImplementedError(f'The model {name} is not currently supported. Please add its loading to this file.')
