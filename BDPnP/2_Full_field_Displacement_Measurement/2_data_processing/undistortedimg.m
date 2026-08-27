clc;
clear;

% Load the parameters
load('cameraParams.mat');
% Load the image
img = imread("Image_2048_1536_1819.bmp");
% Undistort the image
img_undistorted = undistortImage(img,cameraParams); 
% Detect the corner points
[points, boardSize] = detectCheckerboardPoints(img_undistorted);

squareSize = 100;
worldPoints = generateCheckerboardPoints(boardSize, squareSize);
worldPoints(:,1) = worldPoints(:,1) + points(1,1);
worldPoints(:,2) = worldPoints(:,2) + points(1,2);

% Calculate the homography matrix
tform = fitgeotrans(points, worldPoints, 'projective');

% Set the input and output folder paths
input_folder = 'data';
output_folder = 'data_undistorted';

% Create the output folder if it does not exist
if ~isfolder(output_folder)
    mkdir(output_folder);
end

% Get all subfolders in the input folder
subfolders = dir(input_folder);
subfolders = subfolders([subfolders.isdir] & ~ismember({subfolders.name}, {'.', '..'})); % ���˵� . �� ..

% Loop through each subfolder
for i = 1:length(subfolders)
    subfolder_path = fullfile(input_folder, subfolders(i).name);
    
    % Get all image files (assuming PNG format; modify as needed) in the subfolder
    image_files = dir(fullfile(subfolder_path, '*.png')); 
    
    % Create the corresponding output subfolder
    output_subfolder = fullfile(output_folder, subfolders(i).name);
    if ~exist(output_subfolder, 'dir')
        mkdir(output_subfolder);
    end
    
    % Loop through each image file
    for j = 1:length(image_files)
        image_path = fullfile(subfolder_path, image_files(j).name);
        
        % Read the image
        img = imread(image_path);
        % Create a blank image with a resolution of 1536x2048
        blank_image = uint8(zeros(1536, 2048)); 
        % Calculate the image insertion position, assuming it is inserted at the top-left corner
        blank_image(39:1498,51:2002) = img;
        
        img_undistorted = undistortImage(blank_image,cameraParams); 

        % Use imwarp for perspective transformation
        img_transformed = imwarp(img_undistorted, tform, 'OutputView', imref2d(size(img_undistorted)));
        
        save_img = img_transformed(201:1224,401:1424);
        imshow(save_img);
        
        % Save the padded images to the output folder
        output_image_path = fullfile(output_subfolder, image_files(j).name);
        imwrite(save_img, output_image_path);
    end
end
