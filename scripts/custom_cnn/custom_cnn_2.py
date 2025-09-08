from torch import nn 

class CustomCNN_2(nn.Module):
    def __init__(self, fc_layer_size, dropout_rate, num_classes):
        super(CustomCNN_2, self).__init__()
        # define pool and activation func
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
        # first convolutional block
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        
        # second convolutional block
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.25)
        
        # third convolutional block
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.25)

        # fourth convolutional block
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(0.25)
        
        # fully connected layers
        self.fc1 = nn.Linear(50176, fc_layer_size)  
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_layer_size, num_classes)
        
    def forward(self, x):
        # first conv -> batch norm -> pool (224 -> 112) -> dropout (0.25)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        # second conv -> batch norm -> pool (112 -> 56) -> dropout (0.25)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        # third conv -> batch norm -> pool (56 -> 28) -> dropout (0.25)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        # fourth conv -> batch norm -> pool (28 -> 14) -> dropout (0.25)
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        
        # flatten output 
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        
        return x