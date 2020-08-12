import torch 

class mymodel(torch.nn.Module):

  def blocktype1(self, in_channels, out_channels, kernel=5):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=in_channels, padding=2, stride=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=out_channels, stride=1, padding=2),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU()
    )
    return block

  def blocktype2(self, in_channels, out_channels, kernel=3):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=in_channels, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=in_channels, stride=1, padding=1),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(kernel_size=kernel, in_channels=in_channels, out_channels=out_channels, stride=1, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU()
    )
    return block

  def fclayer(self, in_features, out_features):
    layer = torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features, out_features=out_features),
        torch.nn.BatchNorm1d(out_features),
        torch.nn.ReLU()
    )
    return layer

  def pointwiseconv(self, in_channels, out_channels):
    layer = torch.nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels)
    return layer

  def __init__(self):
    super(mymodel, self).__init__()
    self.conv1 = self.blocktype1(3, 64)
    self.p1 = self.pointwiseconv(3,64)
    self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
    self.conv2 = self.blocktype1(64,128)
    self.p2 = self.pointwiseconv(64,128)
    self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
    self.conv3 = self.blocktype2(128,256)
    self.p3 = self.pointwiseconv(128,256)
    self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
    self.conv4 = self.blocktype2(256,512)
    self.p4 = self.pointwiseconv(256,512)
    self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2)
    self.conv5 = self.blocktype2(512,512)
    self.p5 = self.pointwiseconv(512,512)
    self.conv6 = self.blocktype2(512,512)
    self.maxpool5 = torch.nn.MaxPool2d(kernel_size=2)
    self.fc1 = self.fclayer(7*7*512, 64)
    self.fc2 = self.fclayer(64,2)
    self.gradients = None

  def activations_hook(self, grad):
    self.gradients = grad

  def forward(self, x):
    x1 = self.conv1(x)
    x = self.p1(x)
    x+=x1
    x = self.maxpool1(x)
    x1 = self.conv2(x)
    x = self.p2(x)
    x+=x1
    x = self.maxpool2(x)
    x1 = self.conv3(x)
    x = self.p3(x)
    x+=x1
    x = self.maxpool3(x)
    x1 = self.conv4(x)
    x = self.p4(x)
    x+=x1
    x = self.maxpool4(x)
    x1 = self.conv5(x)
    x = self.p5(x)
    x+=x1
    xdash = self.conv6(x)
    #hook = xdash.register_hook(self.activations_hook)  #comment this line when training
    x = self.maxpool5(xdash)
    x = torch.flatten(x, 1)
    fclayer1 = self.fc1(x)
    fclayer2 = self.fc2(fclayer1)
    return fclayer2