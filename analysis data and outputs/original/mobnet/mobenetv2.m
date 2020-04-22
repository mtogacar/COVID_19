
% function DNN_Kod_2
clc; clear all;

tic;

%imge alma
outputFolder = fullfile('D:/OrtakCalismalar/Mesut/COVID/dataset/','fuzzy'); % define output folder
rootFolder = fullfile(outputFolder);
categories = {'covid','normal','pneumo'};


%imds oluþturma
% imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');


tbl = countEachLabel(imds)


 
%%%%%%%%%%%%%%%%%%%% Load pretrained network %%%%%%%%%%%%%%%%%%%%
net =mobilenetv2();
%analyzeNetwork(mobilenetv2())
numel(net.Layers(end).ClassNames)

imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');  




featureLayer = 'Logits';
trainingFeatures = activations(net, trainingSet, featureLayer, ...
    'MiniBatchSize', 1, 'OutputAs', 'columns');

trainingLabels = trainingSet.Labels;


classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');


testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',1);

%predictedLabels = predict(classifier, testFeatures);


testLabels = testSet.Labels;




%%%%%%%%%%%
layersTransfer = net.Layers(1:end-3);
imageSize = net.Layers(1).InputSize

numClasses = (length(categories))
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
numClasses = (length(categories))



if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);
inputSize = net.Layers(1).InputSize
pixelRange = [-30 30];
opts = trainingOptions('sgdm', ...
    'MaxEpochs',50, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'InitialLearnRate',1e-5, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'ValidationData',testSet);

[netTransfer,info] = trainNetwork(imds,lgraph,opts);
%load('netTransfer.mat')

clear classifier

trainingFeatures = activations(netTransfer, trainingSet, 'global_average_pooling2d_1', ...
    'MiniBatchSize', 1, 'OutputAs', 'columns');
 
 
%%%%%%%%%%%%%%%%%%%% Train A Multiclass SVM Classifier Using CNN Features %%%%%%%%%%%%%%%%%%%%

trainingLabels = trainingSet.Labels;

classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');
 
 
%%%%%%%%%%%%%%%%%%%% Evaluate Classifier %%%%%%%%%%%%%%%%%%%%
 
 
% Extract test features using the CNN
testFeatures = activations(netTransfer, testSet, 'global_average_pooling2d_1', ...
    'MiniBatchSize', 1, 'OutputAs', 'columns');
 
% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');
 
% Get the known labels
testLabels = testSet.Labels;
confMat = confusionmat(testLabels, predictedLabels); 

sum(diag(confMat))/sum(confMat(:))

confusionchart(testLabels,predictedLabels);


c = categorical(testLabels)
n = grp2idx(c)

c = categorical(trainingLabels)
n = grp2idx(c)

