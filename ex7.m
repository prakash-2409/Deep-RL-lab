net = alexnet;
net = alexnet('Weights', 'imagenet');
layers = alexnet('Weights', 'none');
deepNetworkDesigner(alexnet)

net = vgg16;
net = vgg16('Weights', 'imagenet');
layers = vgg16('Weights', 'none');
deepNetworkDesigner(vgg16)

net = resnet50;
net = resnet50('Weights', 'imagenet');
lgraph = resnet50('Weights', 'none');
deepNetworkDesigner(resnet50)
