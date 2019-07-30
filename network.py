import math
from torchvision import models as torchvision_models
import torch.nn as nn
from cnn_finetune import make_model


############ The root module: inception v3 with only feature layers #################
class RootInceptionV3(nn.Module):
    def __init__(self, pool_size, pretrained=True):
        super(RootInceptionV3, self).__init__()
        self.pretrained = pretrained
        self.model = self.inception_v3_model(pool_size=pool_size, num_classes=1000)
        self.features = self.model.features

    def inception_v3_model(self, pool_size, num_classes):
        self.pool = nn.AvgPool2d(kernel_size=pool_size)
        original_model = torchvision_models.inception_v3(pretrained=self.pretrained, transform_input=False)
        finetune_model = make_model('inception_v3', num_classes=num_classes, pool=self.pool, pretrained=self.pretrained)
        self.copy_module_weights(original_model.fc, finetune_model._classifier)
        return finetune_model

    def copy_module_weights(self, from_module, to_module):
        to_module.weight.data.copy_(from_module.weight.data)
        to_module.bias.data.copy_(from_module.bias.data)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x


############ The branch module: Classifiers #################
class ClassifierNet(nn.Module):
    def __init__(self, in_channels, num_class, pool_size=8):
        super(ClassifierNet, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_size)
        classifier = [
            nn.Dropout(0.5),
            nn.Linear(in_features=in_channels, out_features=num_class)]
        self.classifier = nn.Sequential(*classifier)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


############ The network #############################################################
class EyeNet(nn.Module):
    def __init__(self, classes, pool_size1, pool_size2, branch_in, multi_task):
        super(EyeNet, self).__init__()

        self.classes = classes
        self.multi_task = multi_task
        self.features = RootInceptionV3(pool_size=pool_size1)
        self.pool = nn.AvgPool2d(kernel_size=pool_size2)
        self.branches = {}
        self.classifiers = {}
        if self.multi_task:
            for name in classes:
                setattr(self, 'classifiers_' + name,
                        ClassifierNet(in_channels=branch_in, num_class=2, pool_size=pool_size2))
        else:
            self.classifier = ClassifierNet(in_channels=branch_in, num_class=len(self.classes), pool_size=pool_size2)

    def forward(self, x):
        x = self.features(x)
        x_out = []
        if self.multi_task:
            for name in self.classes:
                x_out.append(eval('self.classifiers_%s(x)' % name))
        else:
            x_out = self.classifier(x)
        return x_out