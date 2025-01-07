%--------------------------------------------------------------------------
% Script: Deep Learning Model Training and Prediction for OARs only
% Author: Yoganathan, SA, Radiation Oncology, Hamad Medical Corporation,
% Qatar
% Date: January, 2025
% Description: This script trains multiple deep learning models for 
% segmentation tasks using different architectures. It saves the trained 
% networks and performs predictions on test data.
%--------------------------------------------------------------------------

% List of deep learning network architectures to be used
net_name = {'xception', 'inceptionresnetv2', 'resnet18', 'resnet50', 'mobilenetv2'}';

% Corresponding input image sizes for each network architecture
imageSize = {[299 299 3], [299 299 3], [224 224 3], [224 224 3], [224 224 3]}';

% Path to segmentation labels
segPath = 'Y:\Brain\OAR\HaN-Seg\Train\Seg';

% Path to input images
imagePath = 'Y:\Brain\OAR\HaN-Seg\Train\Img';

% Batch size for training for each network
batch_size = {16, 16, 32, 32, 32}';

% Number of epochs to wait for improvement during validation before stopping
val_patience = {75, 75, 75, 75, 75}';

% Cell array to store trained networks and training information
net_cell = cell(5, 2);

% Loop through each network for training (currently set to train only the first network)
for i = 1:1
    % Close any open figures
    close all
    
    % Display a message indicating the start of training for the current network
    sprintf('Started training for network %d: %s', i, net_name{i})
    
    % Call the custom function to train the model
    [net, info] = Train_Networks(imageSize{i}, batch_size{i}, val_patience{i}, net_name{i}, imagePath, segPath);   
    
    % Generate a filename based on the network name
    fname = sprintf(net_name{i});
    
    % Save the trained network and training information to a file
    save(fname, 'net', 'info')
end

% Save the cell array containing all trained networks and training information
save net_cell net_cell

