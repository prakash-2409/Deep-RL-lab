net = googlenet;
net = googlenet('Weights', 'imagenet');
lgraph = googlenet('Weights', 'none');
deepNetworkDesigner(googlenet)

[X, T] = simpleseries_dataset;
net = layrecnet(1:2, 10);
[Xs, Xi, Ai, Ts] = preparets(net, X, T);
net = train(net, Xs, Ts, Xi, Ai);
view(net)
