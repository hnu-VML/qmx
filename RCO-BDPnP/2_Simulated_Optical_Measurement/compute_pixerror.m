clc;
clear;

target = ones(256,256);

% Specify the folder path
folderPath = 'DIC-Result';

% Get a list of all .mat files in the folder
matFiles = dir(fullfile(folderPath, '*.mat'));

for i = 1:length(matFiles)
    matFilePath = fullfile(folderPath, matFiles(i).name);
    load(matFilePath);

    A = data_dic_save.displacements(1).plot_u_dic(39:218,39:218);
    B = 35*target(39:218,39:218);
    
    % Ensure that the corresponding elements in both matrices are non-zero
    mask = (A ~= 0) & (B ~= 0);
    
    % Use a mask to filter out the zero elements
    A_filtered = A(mask);
    B_filtered = B(mask);
    
    error = mean(abs(A_filtered-B_filtered),"all");
    
    % Print the file name and the calculation result
    fprintf('File: %s, Average Error: %.4f pixel\n', matFiles(i).name, abs(error));
end