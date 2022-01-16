import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, channels_img=1, features_d=8, num_classes=6):
        super(Net, self).__init__()
        self.feature_extractor = self._features_layer(channels_img, features_d)
        self.classifier = self._clf_layer(num_classes)

    def _features_layer(self, channels_img, features_d):
        features = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # output: N x features_d*8 x 4 x 4
        )
        return features

    def _clf_layer(self, num_classes):
        classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )   
        return classifier   

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits

if __name__ == '__main__':
    N, in_channels, H, W = 1, 3, 64, 64
    x = torch.randn((N, in_channels, H, W))
    model = Net(in_channels, 8, 5)
    print(model(x).shape)