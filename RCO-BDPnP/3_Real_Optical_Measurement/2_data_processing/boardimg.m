clc;
clear;

% Set the input and output folder paths
input_folder = 'data';
output_folder = 'data_out';

% Create the output folder if it does not exist
if ~isfolder(output_folder)
    mkdir(output_folder);
end

% Get all subfolders in the input folder
subfolders = dir(input_folder);
subfolders = subfolders([subfolders.isdir] & ~ismember({subfolders.name}, {'.', '..'}));

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
        
        % To insert the image at the top-left corner, the insertion position will be (0, 0), which corresponds to the starting point of the blank image
        blank_image(39:1498,51:2002) = img;
        
        % imshow(blank_image);
        
        % Save the filled image to the output folder
        output_image_path = fullfile(output_subfolder, image_files(j).name);
        imwrite(blank_image, output_image_path);
    end
end
