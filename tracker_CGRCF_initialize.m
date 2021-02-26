function [state, location, params] = tracker_CGRCF_initialize(im, region, params)

  if size(im,3)==3
    gray = double(rgb2gray(im));
else 
    gray = im;
end
    
    [height, width] = size(gray);

    % If the provided region is a polygon ...
    if numel(region) > 4
        x1 = round(min(region(1:2:end)));
        x2 = round(max(region(1:2:end)));
        y1 = round(min(region(2:2:end)));
        y2 = round(max(region(2:2:end)));
        region = round([x1, y1, x2 - x1, y2 - y1]);
    else
        region = round([round(region(1)), round(region(2)), ... 
        round(region(1) + region(3)) - round(region(1)), ...
        round(region(2) + region(4)) - round(region(2))]);
    end;
       params.rect_anno(1)=region(1,1);
   params.rect_anno(2)=region(1,2);
     params.rect_anno(3)=region(1,3);
     params.rect_anno(4)=region(1,4);
    
    x1 = max(0, region(1));
    y1 = max(0, region(2));
    x2 = min(width-1, region(1) + region(3) - 1);
    y2 = min(height-1, region(2) + region(4) - 1);
    template = gray((y1:y2)+1, (x1:x2)+1);
    state = struct('template', template, 'size', [x2 - x1 + 1, y2 - y1 + 1]);
    state.window = max(state.size) * 2;
    state.position = [x1 + x2 + 1, y1 + y2 + 1] / 2;

    location = [x1, y1, state.size];

    if state.size(1)<16, state.size(1)=16;end 
    if state.size(2)<16, state.size(2)=16;end 
    
    target_sz = state.size([2,1]); 
    state.pos = state.position([2,1]); 
global enableGPU;
enableGPU = true;
%   HOG feature parameters
params.hog_params.nDim   = 31;
params.cn_params.nDim    =10;
params.cn_params.tablename = 'CNnorm';
params.cn_params.useForGray = false;
%   Global feature parameters 
params.t_features = {
    ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...  % Grayscale is not used as default
    struct('getFeature',@get_fhog,'fparams',params.hog_params),...
%     struct('getFeature',@get_table_feature, 'fparams',cn_params),...
    %struct('getFeature',@prevggmfeature,'fparams',prevggm_params),
    };
params.t_global.cell_size = 4;                  % Feature cell size
params.t_global.cell_selection_thresh = 0.75^2; % Threshold for reducing the cell size in low-resolution cases

%   Search region + extended background parameters
params.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale = 5;           % the size of the training/detection area proportional to the target size
params.search_area_scale_small_target = 6.5;   % % the size of the training/detection area proportional to the small target size
params.filter_max_area   = 50^2;        % the size of the training/detection area in feature grid cells

%   Learning parameters
params.learning_rate       = 0.0185;        % learning rate
params.output_sigma_factor = 1/16;		% standard deviation of the desired correlation output (proportional to target)

%   Detection parameters
params.interpolate_response  = 4;        % correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
params.newton_iterations     = 5;           % number of Newton's iteration to maximize the detection scores
				% the weight of the standard (uniform) regularization, only used when params.use_reg_window == 0
%   Scale parameters
params.number_of_scales =  5;
params.scale_step       = 1.01;
state.frame=1;
%   size, position, frames initialization
params.wsize    = [params.rect_anno(1,4), params.rect_anno(1,3)];
params.init_pos = [params.rect_anno(1,2), params.rect_anno(1,1)] + floor(params.wsize/2);
%   ADMM parameters, # of iteration, and lambda- mu and betha are set in
%   the main function.
params.pe=[1,1,1];
params.admm_iterations = 2;
params.al_iteration = 2;
params.admm_3frame =32 ;
params.admm_lambda = 0.01;
params.admm_lambda1 = 1.2;
params.admm_lambda2 = 0;%0.001; %0
params.admm_beta=0.1;
params.alphaw = 1;
params.alphaq = 1;
params.gamma = 0;
params.ifcompress=1;
state.ifcompress=1;
params.w_init = 7;
%   Debug and visualization
params.visualization = 1;
params.show_regularization = 1;

    params.init_pos = state.pos; 
    params.wsize = target_sz; 

    randn('seed',0);rand('seed',0);

%initialize parts
state.frame = 1;
        
    state.pos = floor(params.init_pos);
    state.target_sz = floor(params.wsize);


end
