from torch import nn 

class CustomCNN_3(nn.Module):
    def __init__(self, fc_layer_size, dropout_rate, num_classes):
        super(CustomCNN_3, self).__init__()

        # v2.F doesn't work so define activation and pool as global
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        # first convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # second convolutional block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # third convolutional block
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # fourth convolutional block
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # fully connected layers + dropout
        self.fc1 = nn.Linear(256 * 14 * 14, fc_layer_size)
        self.dropout = nn.Dropout(dropout_rate)  
        self.fc2 = nn.Linear(fc_layer_size, num_classes)

    def forward(self, x):
        # first conv -> batch norm -> second conv -> batch norm -> pool (224 -> 112)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)

        # third conv -> batch norm -> fourth conv -> batch norm -> pool (112 -> 56)
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.bn2(self.conv4(x)))
        x = self.pool(x)

        # fifth conv -> batch norm -> sixth conv -> batch norm -> pool (56 -> 28)
        x = self.relu(self.bn3(self.conv5(x)))
        x = self.relu(self.bn3(self.conv6(x)))
        x = self.pool(x)

        # seventh conv -> batch norm -> eighth conv -> batch norm -> pool (28 -> 14)
        x = self.relu(self.bn4(self.conv7(x)))
        x = self.relu(self.bn4(self.conv8(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x