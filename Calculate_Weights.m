function dice_weight = Calculate_Weights()
%--------------------------------------------------------------------------
% Author: Yoganathan, SA, Radiation Oncology, Hamad Medical Corporation,
% Qatar
% Date: January, 2025
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% Calculate Dice similarity-based weights for multiple pre-trained networks
% across various segmentation classes (8 classes in total).
%
% Output:
%   - dice_weight: weights of each segment 
%--------------------------------------------------------------------------

% Define network names and their input image sizes
net_name = {'xception','inceptionresnetv2','resnet18','resnet50','mobilenetv2'}';
im_size = {[299 299 3],[299 299 3],[224 224 3],[224 224 3],[224 224 3]}';

%% Calculate weights
% Initialize storage for Dice values (5 networks x 8 classes)
dice_val = zeros(5, 8);

% Loop through each network
for n = 1:5
    imageSize = im_size{n}; % Get the image size for the current network

    % Load test images
    D = 'Y:\Brain\OAR\HaN-Seg\Weight_Opt\Img'; % Directory of weight-calc images
    S = dir(fullfile(D, '*.mat')); % Get all .mat files in the directory
    N = natsortfiles({S.name}); % Sort filenames naturally
    F = cellfun(@(n) fullfile(D, n), N, 'uni', 0); % Generate full file paths
    voldsTest = imageDatastore(F, 'FileExtensions', '.mat', ...
        'ReadFcn', @(x) matRead(x, imageSize)); % Create datastore for test images

    % Initialize arrays for predicted segmentations and original images
    pred_seg = zeros(imageSize); 
    mr = zeros(imageSize);

    % Load the current network
    fname = sprintf(net_name{n}); % Network file name
    load(fname); % Load the network

    % Predict segmentations for each image in the test set
    for i = 1:length(voldsTest.Files)
        slice = read(voldsTest); % Read an image slice
        pred_seg(:, :, i) = semanticseg(slice, net); % Perform semantic segmentation
        mr(:, :, i) = slice(:, :, 1); % Save the first channel (MR image)
    end
    pred_seg = double(pred_seg); % Convert segmentation predictions to double

    % Load reference segmentations
    D = 'Y:\Brain\OAR\HaN-Seg\Weight_Opt\Seg'; % Directory of reference labels
    S = dir(fullfile(D, '*.mat')); % Get all .mat files in the directory
    N = natsortfiles({S.name}); % Sort filenames naturally
    F = cellfun(@(n) fullfile(D, n), N, 'uni', 0); % Generate full file paths
    voldsTest = imageDatastore(F, 'FileExtensions', '.mat', ...
        'ReadFcn', @(x) matReadlabel(x, imageSize)); % Create datastore for labels

    % Initialize array for reference segmentations
    ref_seg = zeros(imageSize);

    % Read each reference segmentation
    for i = 1:length(voldsTest.Files)
        ref_seg(:, :, i) = read(voldsTest); % Read reference segmentation
    end
    ref_seg = double(ref_seg); % Convert reference segmentations to double

    % Calculate Dice coefficients for the current network
    dice_val(n, :) = dice_calc(pred_seg, ref_seg); % Store Dice values for 8 classes
end

% Convert the Dice values into weights
dice_weight = zeros(5, 8); % Initialize weight matrix
for n = 1:5
    for m = 1:8
        dice_weight(n, m) = dice_val(n, m) / sum(dice_val(1:end, m)); % Normalize Dice values
    end
end
end

%% Helper Functions

% Function to read and resize input image
function new_data = matRead(filename, imageSize2)
% Load the .mat file and resize the data to the required image size
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
for d = 1:size(data, 3)
    new_data(:, :, d) = imresize(data(:, :, d), ...
        [imageSize2(1), imageSize2(2)], 'Method', 'nearest');
end
% Create 3 channels (MR data replicated across channels)
new_data(:, :, 1) = new_data(:, :, 1); 
new_data(:, :, 2) = new_data(:, :, 1);
new_data(:, :, 3) = new_data(:, :, 1);
% Normalize the data to the range [-0.5, 0.5]
new_data = rescale(new_data, -0.5, 0.5);
end

% Function to read and resize reference label
function data = matReadlabel(filename, imageSize2)
% Load the .mat file and resize the segmentation label to the required size
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
data = imresize(data, [imageSize2(1), imageSize2(2)], 'Method', 'nearest');
data = data + 1; % Increment labels to avoid zero-indexing
end

% Function to calculate Dice coefficients for 8 classes
function DICE = dice_calc(pred_seg, ref_seg)
% Adjust segmentations to avoid zero-indexing
pred_seg = pred_seg - 1;
ref_seg = ref_seg - 1;

% Initialize ground truth and predicted masks for all 8 classes
bs_g = (ref_seg == 1); oc_g = (ref_seg == 2);
onR_g = (ref_seg == 3); onL_g = (ref_seg == 4);
cocR_g = (ref_seg == 5); cocL_g = (ref_seg == 6);
eyeR_g = (ref_seg == 7); eyeL_g = (ref_seg == 8);

bs_t = (pred_seg == 1); oc_t = (pred_seg == 2);
onR_t = (pred_seg == 3); onL_t = (pred_seg == 4);
cocR_t = (pred_seg == 5); cocL_t = (pred_seg == 6);
eyeR_t = (pred_seg == 7); eyeL_t = (pred_seg == 8);

% Compute Dice coefficients for each class
dice_bs = dice(bs_t, bs_g);
dice_oc = dice(oc_t, oc_g);
dice_onR = dice(onR_t, onR_g);
dice_onL = dice(onL_t, onL_g);
dice_cocR = dice(cocR_t, cocR_g);
dice_cocL = dice(cocL_t, cocL_g);
dice_eyeR = dice(eyeR_t, eyeR_g);
dice_eyeL = dice(eyeL_t, eyeL_g);

% Combine Dice values into a single array
DICE = [dice_bs, dice_oc, dice_onR, dice_onL, ...
        dice_cocR, dice_cocL, dice_eyeR, dice_eyeL];
end
