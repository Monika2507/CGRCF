function params = set_parameters()
% 	params.padding = 3;  %extra area surrounding the target
	params.lambda = 1e-4;  %regularization
	params.output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)
	params.scale_est_mode = 1; %a951106 predefined values: 0=no_scaling 1=our_scaling 2=RPT_scaling 3=GroundTruth-guided_scaling 4=affine_estimation_scaling
     
end
