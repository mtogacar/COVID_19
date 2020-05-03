%imge alma
outputFolder = fullfile('D:/OrtakCalismalar/Mesut/COVID/dataset/','stacking'); % Firat University
rootFolder = fullfile(outputFolder);
categories = {'covid','normal','pneumo'};


%imds oluþturma
% imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');


tbl = countEachLabel(imds)

net =mobilenetv2();
numel(net.Layers(end).ClassNames)

imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');  




featureLayer = 'Logits';
trainingFeatures = activations(net, trainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');

trainingLabels = trainingSet.Labels;


classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');


testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',64);

%predictedLabels = predict(classifier, testFeatures);


testLabels = testSet.Labels;




%%%%%%%%%%%
layersTransfer = net.Layers(1:end-3);
imageSize = net.Layers(1).InputSize

numClasses = (length(categories))

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
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'InitialLearnRate',1e-5, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'ValidationData',testSet);

[netTransfer] = trainNetwork(imds,lgraph,opts);
