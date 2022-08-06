
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

    
class residual_block(nn.Module):
    def __init__(self, in_channels,out_channels,strides = 1):
        super(residual_block, self).__init__()
 
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=strides, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,stride=1 ,padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, padding=0, bias=False),
                torch.nn.BatchNorm2d(out_channels)                
            )
        else:
            self.downsample = None
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ReLU(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.downsample(identity)
        
        x = x + identity
        x = self.ReLU(x)

        return x
    


class myResnet(nn.Module):
    def __init__(self, num_out=10,block=residual_block, num_layers=[2,2,2,2]):
        super(myResnet, self).__init__()
        

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1=self._make_layer(block, 64, 64, num_layers[0], 1)
        self.layer2=self._make_layer(block, 64, 128, num_layers[1], 2)
        self.layer3=self._make_layer(block, 128, 256, num_layers[2], 2)
        self.layer4=self._make_layer(block, 256 , 512, num_layers[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None # nn.Linear(512, num_out)
    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    def freeze_all(self):
        for name, p in self.named_parameters():
            if 'fc' not in name:
                p.requires_grad = False
    def unfreeze_all(self):
        for name, p in self.named_parameters():
            if 'fc' not in name:
                p.requires_grad = True

    def _make_layer(self,block, in_channel, out_channel, num_layers,stride=1):
        layers = [block(in_channel, out_channel,stride)]
        in_channel=out_channel
        for _ in range(num_layers-1):
            layer = block(in_channel, out_channel)
            layers.append(layer)

        
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.ReLU(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

