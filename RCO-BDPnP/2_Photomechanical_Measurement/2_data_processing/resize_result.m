clc;
clear;

targetSize = [256 256];
rootDir = "data_undistorted";
outputDir = "resize_data";

if ~exist(outputDir, 'dir')
    % If the output directory does not exist, create it
    mkdir(outputDir);
end

entries = dir(fullfile(rootDir, '*'));
    
% Loop through all the entries
for k = 1:length(entries)
    entry = entries(k);

    % If it is a directory and not '.' or '..'
    if ~strcmp(entry.name,'.') && ~strcmp(entry.name,'..')
        subFolderPath = fullfile(rootDir, entry.name);
        % Print the current directory path.
        disp(subFolderPath);
        
        suboutputDir = fullfile(outputDir, entry.name);
        if ~exist(suboutputDir, 'dir')
        % If the output directory does not exist, create it
            mkdir(suboutputDir);
        end
        
        % Get a list of all image files in the folder
        filelist = dir(fullfile(subFolderPath, '*.png'));

        % Loop through and read each image
        for i = 1:numel(filelist)
            % Construct the full path for each image
            filename = fullfile(subFolderPath, filelist(i).name);
            % Read the image
            img = imread(filename);

            % Resize the image to 256x256
            resizedImg = imresize(img, targetSize);

            % Save the resized image to the new location
            imwrite(resizedImg, fullfile(suboutputDir, filelist(i).name));
        end
    end
end