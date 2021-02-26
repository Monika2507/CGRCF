function [seq, ground_truth,img_path] = load_video_info(video_path,videoname)


ground_truth = dlmread([video_path  videoname '_gt.txt']);
seq.format = 'otb';
seq.len = size(ground_truth, 1);
seq.init_rect = ground_truth(1,:);

img_path = [video_path 'img/'];

img_files = dir(fullfile(img_path, '*.jpg'));
for i=1:seq.len
img_file{i} = [img_path img_files(i).name];
end

seq.s_frames = (img_file);
end

