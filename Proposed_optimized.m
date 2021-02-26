% This function implements the ASRCF tracker.

function [state, location] = Proposed_optimized(state, im, params)
%   Setting parameters for local use.
see_q = [];
state.search_area_scale   = params.search_area_scale;
output_sigma_factor = params.output_sigma_factor;
state.learning_rate       = params.learning_rate;
filter_max_area     = params.filter_max_area;
nScales             = params.number_of_scales;
scale_step          = params.scale_step;
interpolate_response = params.interpolate_response;
alphaw=params.alphaw;
alphaq=params.alphaq;
update_interval=2;
features    = params.t_features;
state.target_sz   = floor(params.wsize);


% visualization  = params.visualization;
state.init_target_sz = state.target_sz;
pe=params.pe;
if state.frame==1
    if state.init_target_sz(1)*state.init_target_sz(2)<900&&(size(im,1)*size(im,2))/(state.init_target_sz(1)*state.init_target_sz(2))>180
        params.state.search_area_scale=6.5;
        params.pe=[0.1,1,0];
        params.admm_3frame = 0;
        state.ifcompress=0;
    end
    
    state.featureRatio = params.t_global.cell_size;
    state.search_area_pos = prod(state.init_target_sz / state.featureRatio * state.search_area_scale);
    
    % when the number of cells are small, choose a smaller cell size
    if isfield(params.t_global, 'cell_selection_thresh')
        if state.search_area_pos < params.t_global.cell_selection_thresh * filter_max_area
            params.t_global.cell_size = min(state.featureRatio, max(1, ceil(sqrt(prod(state.init_target_sz * state.search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
            
            state.featureRatio = params.t_global.cell_size;
            state.search_area_pos = prod(state.init_target_sz / state.featureRatio * state.search_area_scale);
        end
    end
    
    state.global_feat_params = params.t_global;
    
    if state.search_area_pos > filter_max_area
        state.currentScaleFactor = sqrt(state.search_area_pos / filter_max_area);
    else
        state.currentScaleFactor = 1.0;
    end
    
    % target size at the initial scale
    state.base_target_sz = state.target_sz / state.currentScaleFactor;
    
    % window size, taking padding into account
    switch params.search_area_shape
        case 'proportional'
            state.sz = floor( state.base_target_sz * state.search_area_scale);     % proportional area, same aspect ratio as the target
        case 'square'
            state.sz = repmat(sqrt(prod(state.base_target_sz * state.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
        case 'fix_padding'
            state.sz = state.base_target_sz + sqrt(prod(state.base_target_sz * state.search_area_scale) + (state.base_target_sz(1) - state.base_target_sz(2))/4) - sum(state.base_target_sz)/2; % const padding
        otherwise
            error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
    end
    
    % set the size to exactly match the cell size
    state.sz = round(state.sz / state.featureRatio) * state.featureRatio;
    state.use_sz = floor(state.sz/state.featureRatio);
    
    % construct the label function- correlation output, 2D gaussian function,
    % with a peak located upon the target
    
    state.output_sigma = sqrt(prod(floor(state.base_target_sz/state.featureRatio))) * output_sigma_factor;
    state.rg           = circshift(-floor((state.use_sz(1)-1)/2):ceil((state.use_sz(1)-1)/2), [0 -floor((state.use_sz(1)-1)/2)]);
    state.cg           = circshift(-floor((state.use_sz(2)-1)/2):ceil((state.use_sz(2)-1)/2), [0 -floor((state.use_sz(2)-1)/2)]);
    [state.rs, state.cs]     = ndgrid( state.rg,state.cg);
    state.y            = exp(-0.5 * (((state.rs.^2 + state.cs.^2) / state.output_sigma^2)));
    state.yf           = fft2(state.y); %   FFT of y.
    
    
    if interpolate_response == 1
        state.interp_sz = state.use_sz * state.featureRatio;
    else
        state.interp_sz= state.use_sz;
    end
    
    % construct cosine window
    feature_sz_cell={state.use_sz,state.use_sz,state.use_sz};
    sz=state.sz;
    cos_window = cellfun(@(sz) single(hann(sz(1)+2)*hann(sz(2)+2)'), feature_sz_cell, 'uniformoutput', false);
    state.cos_window = cellfun(@(cos_window) cos_window(2:end-1,2:end-1), cos_window, 'uniformoutput', false);
    cos_window=state.cos_window;
    if state.frame==1
        if size(im,3) == 3
            if all(all(im(:,:,1) == im(:,:,2)))
                colorImage = false;
            else
                colorImage = true;
            end
        else
            colorImage = false;
        end
    end
    % compute feature dimensionality
    feature_dim = 0;
    for n = 1:length(features)
        
        if ~isfield(features{n}.fparams,'useForColor')
            features{n}.fparams.useForColor = true;
        end
        
        if ~isfield(features{n}.fparams,'useForGray')
            features{n}.fparams.useForGray = true;
        end
        
        if (features{n}.fparams.useForColor && colorImage) || (features{n}.fparams.useForGray && ~colorImage)
            feature_dim = feature_dim + features{n}.fparams.nDim;
        end
    end
    
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
    
    if nScales > 0
        scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
        state.scaleFactors = scale_step .^ scale_exp;
        state.min_scale_factor = scale_step ^ ceil(log(max(5 ./ state.sz)) / log(scale_step));
        state.max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ state.base_target_sz)) / log(scale_step));
    end
    
    if interpolate_response >= 3
        % Pre-computes the grid that is used for socre optimization
        state.ky = circshift(-floor((state.use_sz(1) - 1)/2) : ceil((state.use_sz(1) - 1)/2), [1, -floor((state.use_sz(1) - 1)/2)]);
        state.kx = circshift(-floor((state.use_sz(2) - 1)/2) : ceil((state.use_sz(2) - 1)/2), [1, -floor((state.use_sz(2) - 1)/2)])';
        newton_iterations = params.newton_iterations;
    end
    
    time = 0;
    
    % allocate memory for multi-scale tracking
    multires_pixel_template = zeros(state.sz(1), state.sz(2), size(im,3), nScales, 'uint8');
    state.small_filter_sz = floor(state.base_target_sz/state.featureRatio);
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
end
tic();
%% main loop
if state.frame > 1
    for scale_ind = 1:nScales
        multires_pixel_template(:,:,:,scale_ind) = ...
            get_pixels(im, state.pos, round(state.sz*state.currentScaleFactor*state.scaleFactors(scale_ind)), state.sz);
    end
    
    for scale_ind = 1:nScales
        xt_hc(:,:,:,scale_ind)=get_features(multires_pixel_template(:,:,:,scale_ind),features,state.global_feat_params);
        xt_hcf(:,:,:,scale_ind)=fft2(bsxfun(@times,xt_hc(:,:,:,scale_ind),state.cos_window{1}));
    end
    xt=extract_features(multires_pixel_template(:,:,:,3),state.use_sz,features,state.global_feat_params,state.frame,state.ifcompress,params.pe,params);
    xt1=gather(xt{1});
    xt2=gather(xt{2});
    xt3=gather(xt{3});
    xtf{1} = fft2(bsxfun(@times,xt1,state.cos_window{1}));
    xtf{2} = fft2(bsxfun(@times,xt2,state.cos_window{2}));
    xtf{3} = fft2(bsxfun(@times,xt3,state.cos_window{3}));
    xtf=cat(3,xtf{1},xtf{2},xtf{3});
    responsef=permute(sum(bsxfun(@times, conj(state.g_f), xtf), 3), [1 2 4 3]);
    response_hcf=permute(sum(bsxfun(@times, conj(state.g_hcf), xt_hcf), 3), [1 2 4 3]);
    
    
    
    r1=permute(bsxfun(@times, conj(state.g_f), xtf), [1 2 4 3]);
    r2=permute(bsxfun(@times, conj(state.g_hcf), xt_hcf), [1 2 4 3]);
    
    responsef=gather(responsef);
    response_hcf=gather(response_hcf);
    
    % if we undersampled features, we want to interpolate the
    % response so it has the same size as the image patch
    if interpolate_response == 2
        % use dynamic interp size
        state.interp_sz = floor(size(state.y) * state.featureRatio * state.currentScaleFactor);
    end
    responsef_padded = resizeDFT2(responsef, state.interp_sz);
    responsehcf_padded = resizeDFT2(response_hcf, state.use_sz);
    % response in the spatial domain
    response = ifft2(responsef_padded, 'symmetric');
    responsehc = ifft2(responsehcf_padded, 'symmetric');

    if interpolate_response == 3
        error('Invalid parameter value for interpolate_response');
    elseif interpolate_response == 4
        
        [~, ~, sind] = resp_newton(responsehc, responsehcf_padded, params.newton_iterations, state.ky, state.kx, state.use_sz);
        [disp_row, disp_col, ~] = resp_newton(response, responsef_padded, params.newton_iterations, state.ky, state.kx, state.use_sz);
        
    else
        [row, col, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
        disp_row = mod(row - 1 + floor((state.interp_sz(1)-1)/2), state.interp_sz(1)) - floor((state.interp_sz(1)-1)/2);
        disp_col = mod(col - 1 + floor((state.interp_sz(2)-1)/2), state.interp_sz(2)) - floor((state.interp_sz(2)-1)/2);
    end
    % calculate translation
    switch interpolate_response
        case 0
            translation_vec = round([disp_row, disp_col] * state.featureRatio * state.currentScaleFactor * state.scaleFactors(sind));
        case 1
            translation_vec = round([disp_row, disp_col] * state.currentScaleFactor * state.scaleFactors(sind));
        case 2
            translation_vec = round([disp_row, disp_col] * state.scaleFactors(sind));
        case 3
            translation_vec = round([disp_row, disp_col] * state.featureRatio * state.currentScaleFactor * state.scaleFactors(sind));
        case 4
            translation_vec = round([disp_row, disp_col] * state.featureRatio * state.currentScaleFactor * state.scaleFactors(sind));
    end
    
    % set the scale
    state.currentScaleFactor = state.currentScaleFactor * state.scaleFactors(sind);
    % adjust to make sure we are not to large or to small
    if state.currentScaleFactor < state.min_scale_factor
        state.currentScaleFactor = state.min_scale_factor;
    elseif state.currentScaleFactor > state.max_scale_factor
        state.currentScaleFactor = state.max_scale_factor;
    end
    
    % update position
    state.old_pos = state.pos;
    state.pos = state.pos + translation_vec;
    if state.pos(1)<0||state.pos(2)<0||state.pos(1)>size(im,1)||state.pos(2)>size(im,2)
        state.pos=state.old_pos;
        state.learning_rate=0;
    end
end
state.target_sz = floor(state.base_target_sz * state.currentScaleFactor);

%save position and calculate FPS
state.rect_position(state.frame,:) = [state.pos([2,1]) - floor(state.target_sz([2,1])/2), state.target_sz([2,1])];



if state.frame==1
    % extract training sample image region
    state.pixels = get_pixels(im,state.pos,round(state.sz*state.currentScaleFactor),state.sz);
    state.pixels = uint8(gather(state.pixels));
    x=extract_features(state.pixels,state.use_sz,features,state.global_feat_params,state.frame,state.ifcompress,params.pe,params);
    xx1=gather(x{1});
    xx2=gather(x{2});
    xx3=gather(x{3});
    xf{1} = fft2(bsxfun(@times,xx1,cos_window{1}));
    xf{2} = fft2(bsxfun(@times,xx2,cos_window{2}));
    xf{3} = fft2(bsxfun(@times,xx3,cos_window{3}));
    xf=cat(3,xf{1},xf{2},xf{3});
else
    % use detection features
    shift_samp_pos = 2*pi * translation_vec ./ (state.scaleFactors(sind)*state.currentScaleFactor * state.sz);
    xf = shift_sample(xtf, shift_samp_pos, state.kx', state.ky');
end
xhcf=xf(:,:,1:31);

if (state.frame == 1)
    state.model_xf =xf;
    state.model_xhcf=xhcf;
    state.model_w=gpuArray(construct_regwindow(state.use_sz,state.small_filter_sz));
    
    state.channel_weight=ones(size(state.model_xf)); %AT
elseif state.frame==1||mod(state.frame,update_interval)==0
    state.model_xf = ((1 - state.learning_rate) * state.model_xf) + (state.learning_rate * xf);
    state.model_xhcf = ((1 - state.learning_rate) * state.model_xhcf) + (state.learning_rate * xhcf);
end

% ADMM solution
if (state.frame==1||mod(state.frame,update_interval)==0)
    state.w = gpuArray(params.w_init*single(ones(state.use_sz)));
    L=Graph_L(xf,state);
    % ADMM solution for localization
    [state.g_f,state.h_f,state.channel_weight]=ADMM_solve_h(params,state.use_sz,state.model_xf,state.yf,state.small_filter_sz,state.w,state.model_w,state.frame,state.channel_weight,L);
    for iteration = 1:params.al_iteration-1
        [state.w]=ADMM_solve_w(params,state.use_sz,state.model_w,state.h_f);
        [state.g_f,state.h_f,state.channel_weight]=ADMM_solve_h(params,state.use_sz,state.model_xf,state.yf,state.small_filter_sz,state.w,state.model_w,state.frame,state.channel_weight,L);
    end
    state.model_w=gather(state.model_w);
    state.w=gather(state.w);
    state.model_w=alphaw*state.w+(1-alphaw)*state.model_w;
    % ADMM solution for scale estimation
    [state.g_hcf]=ADMM_base(params,state.use_sz,state.model_xhcf,xhcf,state.yf,state.small_filter_sz,state.frame);
    
end


%     time = time + toc();
rect_position_vis = [state.pos([2,1]) - state.target_sz([2,1])/2, state.target_sz([2,1])];
figure(1)
imshow(im)
hold on
rectangle('Position',rect_position_vis)
hold on

state.position = state.pos([2,1]);
state.size = state.target_sz([2,1]);
location = [state.position - state.size/2, state.size];
location = str2num(num2str(location)); %to debug the line code 120: " traxserver('status', region); " in vot.m file

end
