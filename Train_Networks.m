function [net, info] = Train_Networks(imageSize, batch_size, val_patience, network_name, imagePath, segPath)
%--------------------------------------------------------------------------
% Author: Yoganathan, SA, Radiation Oncology, Hamad Medical Corporation,
% Qatar
% Date: January, 2025
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% Function: fun_oar_deeplearning
% Description: Trains a deep learning model for segmentation using the 
% specified network architecture and training data.
%
% Inputs:
%   - imageSize: Dimensions of input images (e.g., [224 224 3])
%   - batch_size: Mini-batch size for training
%   - val_patience: Early stopping patience during validation
%   - network_name: Name of the network architecture (e.g., 'resnet50')
%   - imagePath: Path to input image data
%   - segPath: Path to corresponding segmentation labels
%
% Outputs:
%   - net: Trained deep learning network
%   - info: Training progress information
%--------------------------------------------------------------------------

% Add required paths for dependencies or utility functions
addpath 'Y:\Brachy\Final_Program\'

% Load input image data
D = imagePath;
S = dir(fullfile(D, '*.mat')); % List all .mat files in the directory
N = natsortfiles({S.name});    % Sort files naturally
F = cellfun(@(n) fullfile(D, n), N, 'uni', 0); % Full file paths
volReader = @(x) matRead(x, imageSize);       % Define a reader function
volds = imageDatastore(F, 'FileExtensions', '.mat', 'ReadFcn', volReader);

% Load segmentation label data
D = segPath;
S = dir(fullfile(D, '*.mat')); % List all .mat files in the directory
N = natsortfiles({S.name});    % Sort files naturally
F = cellfun(@(n) fullfile(D, n), N, 'uni', 0); % Full file paths

% Define pixel label datastore
pxds = pixelLabelDatastore(F, ...
    ["BK" "BS" "OC" "onR" "onL" "cocL" "cocR" "eyeR" "eyeL"], ...
    [1 2 3 4 5 6 7 8 9], ...
    'FileExtensions', '.mat', 'ReadFcn', @(x) matReadlabel(x, imageSize));

%% Validation and Training Data Preparation
% Partition data into training and validation sets
[imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionData(volds, pxds, imageSize);

% Create pixel label image datastores for training and validation
pximds = pixelLabelImageDatastore(imdsTrain, pxdsTrain);
pximdsVal = pixelLabelImageDatastore(imdsVal, pxdsVal);

%% Define the Network
% Specify the number of classes for segmentation
numClasses = 9;

% Load DeepLabv3+ architecture with specified network and downsampling factor
lgraph = deeplabv3plusLayers(imageSize, numClasses, network_name, "DownsamplingFactor", 16);

%% Handle Class Imbalance
% Compute class weights to address imbalanced data
tbl = countEachLabel(pxdsTrain);                  % Count pixels per class
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount; % Compute frequency
classWeights = median(imageFreq) ./ imageFreq;    % Compute weights

% Replace the pixel classification layer with a custom one using class weights
pxLayer = pixelClassificationLayer('Name', 'labels', 'Classes', tbl.Name, 'ClassWeights', classWeights);
old_name = lgraph.Layers(end, 1).Name;           % Get the name of the last layer
lgraph = replaceLayer(lgraph, old_name, pxLayer); % Replace the layer

%% Training
% Define training options
options = trainingOptions('adam', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 4, ...
    'LearnRateDropFactor', 0.3, ...
    'InitialLearnRate', 0.5e-3, ...
    'L2Regularization', 0.000005, ...
    'ValidationData', pximdsVal, ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', batch_size, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 50, ...
    'Plots', 'training-progress', ...
    'ValidationPatience', val_patience, ...
    'ValidationFrequency', 50, ...
    'ExecutionEnvironment', 'gpu');

% Train the network
[net, info] = trainNetwork(pximds, lgraph, options);
end

%% Helper Function: matRead (Process Image Data)
function new_data = matRead(filename, imageSize2)
% Read and resize image data from .mat files
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
for d = 1:size(data, 3)
    new_data(:, :, d) = imresize(data(:, :, d), [imageSize2(1), imageSize2(2)], 'Method', 'nearest');
end
new_data(:, :, 1) = new_data(:, :, 1); % Use first channel for grayscale image
new_data(:, :, 2) = new_data(:, :, 1); % Duplicate for RGB
new_data(:, :, 3) = new_data(:, :, 1); % Duplicate for RGB
new_data = rescale(new_data, -0.5, 0.5); % Normalize the data
end

%% Helper Function: matReadlabel (Process Label Data)
function data = matReadlabel(filename, imageSize2)
% Read and resize segmentation label data
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
data = imresize(data, [imageSize2(1), imageSize2(2)], 'Method', 'nearest');
data = data + 1; % Adjust labels to be 1-indexed
end

%% Helper Function: partitionData (Split Data)
function [imdsTrain, imdsVal, pxdsTrain, pxdsVal] = partitionData(imds, pxds, imageSize2)
% Split data into training (85%) and validation (15%)

% Set random seed for reproducibility
rng(0);
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Split indices for training and validation
numTrain = round(0.85 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);
valIdx = shuffledIndices(numTrain + 1:end);

% Create datastores for training and validation
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
imdsTrain = imageDatastore(trainingImages, 'FileExtensions', '.mat', 'ReadFcn', @(x) matRead(x, imageSize2));
imdsVal = imageDatastore(valImages, 'FileExtensions', '.mat', 'ReadFcn', @(x) matRead(x, imageSize2));

% Create pixel label datastores
classes = pxds.ClassNames;
labelIDs = [1 2 3 4 5 6 7 8 9];
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs, 'FileExtensions', '.mat', 'ReadFcn', @(x) matReadlabel(x, imageSize2));
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs, 'FileExtensions', '.mat', 'ReadFcn', @(x) matReadlabel(x, imageSize2));
end
