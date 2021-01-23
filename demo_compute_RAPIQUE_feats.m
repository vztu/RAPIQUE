%%
% Compute features for a set of video files from datasets
% 
close all; 
clear;

% add path
addpath(genpath('include'));

%%
% parameters
algo_name = 'RAPIQUE'; % algorithm name, eg, 'V-BLIINDS'
data_name = 'KONVID_1K';  % dataset name, eg, 'KONVID_1K'
write_file = true;  % if true, save features on-the-fly
log_level = 0;  % 1=verbose, 0=quite

if strcmp(data_name, 'KONVID_1K')
    root_path = '/media/ztu/Seagate-ztu-ugc/KONVID_1K/';
    data_path = '/media/ztu/Seagate-ztu-ugc/KONVID_1K/KoNViD_1k_videos';
elseif strcmp(data_name, 'LIVE_VQC')
    root_path = '/media/ztu/Seagate-ztu-ugc/LIVE_VQC/';
    data_path = '/media/ztu/Seagate-ztu-ugc/LIVE_VQC/VideoDatabase';
elseif strcmp(data_name, 'YOUTUBE_UGC')
    root_path = '/media/ztu/Seagate-ztu-ugc/YT_UGC';
    data_path = '/media/ztu/Seagate-ztu-ugc/YT_UGC/original_videos';
end

%%
% create temp dir to store decoded videos
video_tmp = '/media/ztu/Data/tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = 'mos_files';
filelist_csv = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(filelist_csv);
num_videos = size(filelist,1);
out_path = 'feat_files';
if ~exist(out_path, 'dir'), mkdir(out_path); end
out_mat_name = fullfile(out_path, [data_name,'_',algo_name,'_feats.mat']);
feats_mat = [];
feats_mat_frames = cell(num_videos, 1);
%===================================================

% init deep learning models
minside = 512.0;
net = resnet50;
layer = 'avg_pool';

%% extract features
% parfor i = 1:num_videos % for parallel speedup
for i = 1:num_videos
    progressbar(i/num_videos) % Update figure
    if strcmp(data_name, 'KONVID_1K')
        video_name = fullfile(data_path, ...
            [num2str(filelist.flickr_id(i)),'.mp4']);
        yuv_name = fullfile(video_tmp, [num2str(filelist.flickr_id(i)), '.yuv']);
    elseif strcmp(data_name, 'LIVE_VQC')
        video_name = fullfile(data_path, filelist.File{i});
        yuv_name = fullfile(video_tmp, [filelist.File{i}, '.yuv']);
    elseif strcmp(data_name, 'YOUTUBE_UGC')
        video_name = fullfile(data_path, filelist.category{i}, ...
            [num2str(filelist.resolution(i)),'P'],[filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    end
    fprintf('\n\nComputing features for %d sequence: %s\n', i, video_name);

    % decode video and store in temp dir
    cmd = ['ffmpeg -loglevel error -y -i ', video_name, ...
        ' -pix_fmt yuv420p -vsync 0 ', yuv_name];
    system(cmd);  

    % get video meta data
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));

    % calculate video features
    tStart = tic;
    feats_frames = calc_RAPIQUE_features(yuv_name, width, height, ...
        framerate, minside, net, layer, log_level);
    fprintf('\nOverall %f seconds elapsed...', toc(tStart));
    % 
    feats_mat(i,:) = nanmean(feats_frames);
    feats_mat_frames{i} = feats_frames;
    % clear cache
    delete(yuv_name)

    if write_file
        save(out_mat_name, 'feats_mat');
%         save(out_mat_name, 'feats_mat', 'feats_mat_frames');
    end
end




