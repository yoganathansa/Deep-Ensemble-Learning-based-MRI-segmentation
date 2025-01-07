function pred_seg = Weighted_Predictions(imPath, dice_weight)
%--------------------------------------------------------------------------
% Author: Yoganathan, SA, Radiation Oncology, Hamad Medical Corporation,
% Qatar
% Date: January, 2025
%--------------------------------------------------------------------------
% --------------------------------------------------------------------------
% % Description: Combines predictions from multiple trained networks using
% weighted probabilities (based on Dice similarity) to produce a final
% segmentation.
%
% Inputs:
%   - imPath: Path to the test images (stored as .mat files)
%   - dice_weight: Weight matrix (size: num_networks x num_classes) with
%                  Dice similarity scores for each network-class pair
%
% Output:
%   - pred_seg: Combined segmentation result
%--------------------------------------------------------------------------

% Define network names and input image sizes
net_name = {'xception', 'inceptionresnetv2', 'resnet18', 'resnet50', 'mobilenetv2'}';
im_size = {[299 299 3], [299 299 3], [224 224 3], [224 224 3], [224 224 3]}';

%% Load Trained Networks
% Load pre-trained networks and their metadata
for n = 1:5
    fname = sprintf(net_name{n}); % Generate the file name
    net = importdata(fname);      % Load the network file
    net_cell{n, 1} = net.net;     % Store the network in a cell array
    net_cell{n, 2} = net.info;    % Store additional information (if needed)
end

%% Load Test Data and Predict Activations
% Prepare the test image data
D = imPath;
S = dir(fullfile(D, '*.mat'));           % List all .mat files in the directory
N = natsortfiles({S.name});              % Sort the files naturally
F = cellfun(@(n) fullfile(D, n), N, 'uni', 0); % Full file paths

% Initialize activation storage
act_cell = cell(length(net_cell), 1);

% Process images through each network
for i = 1:length(net_cell)
    % Create an image datastore for the network's input size
    ds = imageDatastore(F, 'FileExtensions', '.mat', 'ReadFcn', @(x) matRead(x, im_size{i}));
    
    % Get the current network
    net = net_cell{i, 1};
    
    % Compute softmax activations for the test data
    act = activations(net, ds, 'softmax-out');
    
    % Resize activations if needed
    if i <= 2 % Resize for networks 
        act_cell{i, 1}.mask = imresize(act, [224, 224], 'Method', 'nearest');
    else
        act_cell{i, 1}.mask = act;
    end
    
    % Store the network name for reference
    act_cell{i, 1}.name = sprintf(net_name{i});
end

%% Combine Predictions with Weighted Probabilities
% Initialize probability maps for each class
num_classes = 8; % Excluding background
prob_maps = zeros([size(act_cell{1, 1}.mask, 1), size(act_cell{1, 1}.mask, 2), num_classes]);

% Aggregate class probabilities using Dice weights
for i = 1:5
    act_img = act_cell{i, 1}.mask;
    act_img = permute(act_img, [1 2 4 3]); % Permute dimensions for compatibility    
    for c = 1:num_classes
        prob_maps(:, :, c) = prob_maps(:, :, c) + dice_weight(i, c) .* act_img(:, :, :, c + 1);
    end
end

%% Convert Probabilities to Segmentation
% Apply a threshold to determine final segmentation
threshold = 0.9;
pred_seg = zeros(size(prob_maps, 1), size(prob_maps, 2));

for c = 1:num_classes
    pred_seg(prob_maps(:, :, c) > threshold) = c;
end

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
new_data(:, :, 1) = new_data(:, :, 1); % Only MR (or grayscale) data
new_data(:, :, 2) = new_data(:, :, 1); % Duplicate for RGB compatibility
new_data(:, :, 3) = new_data(:, :, 1); % Duplicate for RGB compatibility
new_data = rescale(new_data, -0.5, 0.5); % Normalize between -0.5 and 0.5
end
