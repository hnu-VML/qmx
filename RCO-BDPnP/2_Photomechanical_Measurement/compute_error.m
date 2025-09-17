clc;
clear;

speed = 360;

target = ones(256,256);
imgratio = (1/256*1024)/2388*(247.6-7.12*2);
ipadratio = 1/2388*(247.6-7.12*2);
k=15;

% Specify the folder path
folderPath = '3_DIC-Result';

% Get a list of all .mat files in the folder
matFiles = dir(fullfile(folderPath, '*.mat'));

for i = 1:length(matFiles)
    matFilePath = fullfile(folderPath, matFiles(i).name);
    load(matFilePath);

    A = data_dic_save.displacements(k).plot_u_dic(39:218,39:218)*imgratio;
    B = speed/40*k*ipadratio*target(39:218,39:218); %65:193,65:193
    
    % Ensure that the corresponding elements in both matrices are non-zero
    mask = (A ~= 0) & (B ~= 0);
    
     % Use a mask to filter out the zero elements
    A_filtered = A(mask);
    B_filtered = B(mask);
    
    error = mean(abs(A_filtered-B_filtered),"all");
    
    % Print the file name and the calculation result
    fprintf('File: %s, Average Error: %.4f mm\n', matFiles(i).name, abs(error));
end