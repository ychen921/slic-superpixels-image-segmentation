import torch.nn as nn
import torchvision.models as models

# Model definition
class SegmentationNN(nn.Module):
    def __init__(self):
        super(SegmentationNN, self).__init__()

        # Load pre-trained model
        self.model = models.vgg19(weights=True)

        for param in self.model.features.parameters():
          param.requires_grad = False

        # Replace last few layers with FCL
        self.model.classifier = nn.Sequential(
                  nn.Linear(25088, 10000),
                  nn.ReLU(),
                  nn.Dropout(0.5),
                  nn.Linear(10000, 500),
                  nn.ReLU(),
                  nn.Dropout(0.5),
                  nn.Linear(500, 14))



    def forward(self, input):
        output = self.model(input)
        return output