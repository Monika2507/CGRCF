clc
clear all
close all

base_path   = './data/';
video='Airport_ce';

% Initialize the tracker
params = set_parameters();
load vggmnet.mat
load vgg16net.mat

params.vggmnet=vggmnet;
params.vgg16net=vgg16net;

video_path = [base_path video '/' ];

[seq, ground_truth,video_path] = load_video_info(video_path,video);
region=seq.init_rect;
names=seq.s_frames;
[state, ~, params] = tracker_CGRCF_initialize(imread(names{1}), region, params);
addpath ./external/matconvnet/matlab
vl_setupnn
state.rect_position = zeros(10, 4);
state.rects = zeros(10, 4);

for i=1:seq.len  
im= imread(names{i});
    [state, region] = CGRCF_optimized(state,im, params);
   state.frame = state.frame + 1;
end
results.res = state.rect_position;
result=round(results.res);
