"""
    Sample model.
    Implements Grad-CAM as well.
"""


import torch

class mymodel(torch.nn.Module):

  def blocktype1(self, in_channels, out_channels, kernel=5):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=in_channels, padding=2, stride=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.Dropout2d(p=0.15),
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=out_channels, padding=2, stride=1 ),
        torch.nn.ReLU()
    )
    return block

  def blocktype2(self, in_channels, out_channels, kernel=3):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=in_channels, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=in_channels, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.Dropout2d(p=0.2),
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=out_channels, stride=1, padding=1),
        torch.nn.ReLU()
    )
    return block

  def blocktype3(self, in_channels, out_channels, kernel=7):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=in_channels, padding=3, stride=1),
        torch.nn.ReLU(), 
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.Dropout2d(p=0.1),
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=out_channels, padding=3, stride=1),
        torch.nn.ReLU()
    )
    return block

  def special_convolutions(self, in_channels, out_channels, step, kernel=3):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel, in_channels = in_channels, out_channels=out_channels, padding=1, stride=2**step)
    )
    return block

  def fclayer(self, in_features, out_features):
    layer = torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features, out_features=out_features),
        torch.nn.BatchNorm1d(num_features=out_features),
        torch.nn.Dropout(p=0.25),
        torch.nn.ReLU()
    )
    return layer

  def pointwiseconv(self, in_channels, out_channels):
    layer = torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels)
    return layer

  def __init__(self):
    super(mymodel, self).__init__()
    self.first = self.pointwiseconv(3,4)
    self.resize1 = self.special_convolutions(4,128,step=1)
    self.resize2 = self.special_convolutions(4,512,step=4)
    self.conv1 = self.blocktype3(4,64)
    self.p1 = self.pointwiseconv(4,64)
    self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)

    self.conv2 = self.blocktype1(64,128)
    self.p2 = self.pointwiseconv(64,128)
    self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

    self.conv3 = self.blocktype1(128,256)
    self.p3 = self.pointwiseconv(128,256)
    self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

    self.conv4 = self.blocktype2(256,256)
    self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2)

    self.conv5 = self.blocktype2(256,512)
    self.p5 = self.pointwiseconv(256,512)
    self.conv6 = self.blocktype2(512,512)
    self.conv7 = self.blocktype2(512,1024)
    self.maxpool5 = torch.nn.MaxPool2d(kernel_size=2)

    self.fc1 = self.fclayer(7*7*1024,64)
    self.fc2 = self.fclayer(256,16)
    self.fc3 = self.fclayer(16,2)


  def activations_hook(self, grad):
    self.gradients = grad


  def forward(self, x):
    x = self.first(x)
    r1 = self.resize1(x)
    r2 = self.resize2(x)
    x1 = self.conv1(x)
    x = self.p1(x)
    x = x+x1
    x = self.maxpool1(x)

    x1 = self.conv2(x)
    x = self.p2(x)
    x = x+x1
    x = x+r1
    x = self.maxpool2(x)

    x1 = self.conv3(x)
    x = self.p3(x)
    x = x+x1
    x = self.maxpool3(x)

    x1 = self.conv4(x)
    x = x+x1
    x = self.maxpool4(x)

    x1 = self.conv5(x)
    x = self.p5(x)
    x = x+x1
    x = x+r2
    x1 = self.conv6(x)
    x = x+x1
    xdash = self.conv7(x)
    #hook = xdash.register_hook(self.activations_hook)  #comment this line when training
    x = self.maxpool5(xdash)
    x = torch.flatten(x, 1)
    fclayer1 = self.fc1(x)
    fclayer2 = self.fc2(fclayer1)
    #fclayer3 = self.fc3(fclayer2)
    return fclayer2


  def get_activations_gradient(self):
    return self.gradients


  def get_activations(self,x):
    x = self.first(x)
    r1 = self.resize1(x)
    r2 = self.resize2(x)
    x1 = self.conv1(x)
    x = self.p1(x)
    x = x+x1
    x = self.maxpool1(x)

    x1 = self.conv2(x)
    x = self.p2(x)
    x = x+x1
    x = x+r1
    x = self.maxpool2(x)

    x1 = self.conv3(x)
    x = self.p3(x)
    x = x+x1
    x = self.maxpool3(x)

    x1 = self.conv4(x)
    x = x+x1
    x = self.maxpool4(x)

    x1 = self.conv5(x)
    x = self.p5(x)
    x = x+x1
    x = x+r2
    x1 = self.conv6(x)
    x = x+x1
    xdash = self.conv7(x)
    return xdash