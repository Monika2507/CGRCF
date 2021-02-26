% GET_FEATURES: Extracting hierachical convolutional features

function feat = get_features_seresnet50(im, cos_window, layers, params)

global net
global enableGPU
% gpuDevice()
if isempty(net)
net = dagnn.DagNN.loadobj(load('SE-ResNet-50-mcn.mat')) ;
net.mode = 'test' ;
end

sz_window = cos_window;

% Preprocessing
img = single(im);        % note: [0, 255] range
img = imResample(img, net.meta.normalization.imageSize(1:2));

net.conserveMemory = false; 
% run the CNN
net.eval({'data', img}) ;

% Initialize feature maps
feat = cell(1, 1);

   x = net.vars(net.getVarIndex(layers)).value ;%net.vars(net.getVarIndex('prob')).value ;
   
    x = imResample(x, sz_window(1:2));
    x = gpuArray(x);

    feat=x;
end


