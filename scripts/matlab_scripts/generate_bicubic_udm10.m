function generate_LR_udm10()
%% matlab code to genetate bicubic-downsampled for Vimeo90K dataset

up_scale = 4;
mod_scale = 4;
idx = 0;
filepaths = dir('/data1/lihao/datasets/VSR/udm10/GT/*/*.png');
for i = 1 : length(filepaths)
    [~,imname,ext] = fileparts(filepaths(i).name);
    %% 0001 .png
    folder_path = filepaths(i).folder;
    %% /data1/lihao/datasets/VSR/udm10/GT/archpeople
    save_LR_folder = strrep(folder_path,'GT', 'BIx4');
    if ~exist(save_LR_folder, 'dir')
        mkdir(save_LR_folder);
    end
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_result = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_result);
        % read image
        img = imread(fullfile(folder_path, [imname, ext]));
        img = im2double(img);
        % modcrop
        img = modcrop(img, mod_scale);
        % LR
        im_LR = imresize(img, 1/up_scale, 'bicubic');
        if exist('save_LR_folder', 'var')
            imwrite(im_LR, fullfile(save_LR_folder, [imname, '.png']));
        end
    end
end
end

%% modcrop
function img = modcrop(img, modulo)
if size(img,3) == 1
    sz = size(img);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2),:);
end
end
