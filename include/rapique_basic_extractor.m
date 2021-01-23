function ftrs = rapique_basic_extractor(img)
ftrs = [];
% MSCN
filtlength = 7;
window = fspecial('gaussian', filtlength, filtlength/6);
window = window/sum(sum(window));
mu = imfilter(img, window, 'replicate');
mu_sq = mu.*mu;
sigma = sqrt(abs(imfilter(img.*img, window, 'replicate') - mu_sq));
struct = (img-mu)./(sigma+1);
%% 2 GGD model parameters
[gamparam, sigparam] = est_GGD_param(struct(:));
ftrs = [ftrs gamparam sigparam]; ...
%     skewness(struct(:)) kurtosis(struct(:))];

%% 2 model parameters
sigmaParam  = nakafit(sigma(:) + eps);
% [lambda, k] = est_WBL_param(sigma(:) + eps);
% sigmaParam  = gamfit(sigma(:)+eps);
% ftrs = [ftrs lambda k sigmaParam(1) sigmaParam(2)];
ftrs = [ftrs sigmaParam(1) sigmaParam(2)];
%     skewness(sigma(:)) kurtosis(sigma(:))];

%% pair-production - AGGD - 4x4 = 16 features
shifts = [0 1; 1 0 ; 1 1; -1 1];
for itr_shift = 1:4
    shifted_structdis = circshift(struct, shifts(itr_shift,:));
    pair = struct(:) .* shifted_structdis(:);
    [alpha, leftstd, rightstd] = est_AGGD_param(pair);
    const = (sqrt(gamma(1/alpha)) / sqrt(gamma(3/alpha)));
    meanparam = (rightstd - leftstd) * (gamma(2/alpha)/gamma(1/alpha)) * const;
    ftrs = [ftrs alpha meanparam leftstd rightstd];               
end

% log 
struct = log(abs(struct) + 0.1);
%% log-derivatives - 7x2 = 14 features
% 4 directions
shifts = [0 1; 1 0 ; 1 1; -1 1];
for itr_shift = 1:4
    structdis_shift_tmp = circshift(struct, shifts(itr_shift, :));
    structdis_diff = struct - structdis_shift_tmp;
    [gamparam, sigma] = est_GGD_param(structdis_diff(:));
    ftrs = [ftrs gamparam sigma];
    structdis_shift(itr_shift) = {[structdis_shift_tmp]};
end
% 5th direction
structdis_diff = struct + structdis_shift{1,3} - structdis_shift{1,1} - structdis_shift{1,2};
[gamparam, sigma ] = est_GGD_param(structdis_diff(:));
ftrs = [ftrs gamparam sigma];
% 6th 7th direction
win_tmp_1 = [0 1 0; -1 0 -1; 0 1 0;];
win_tmp_2 = [1 0 -1; 0 0 0; -1 0 1;];
structdis_diff_1 = imfilter(struct, win_tmp_1, 'replicate');
structdis_diff_2 = imfilter(struct, win_tmp_2, 'replicate');
[gamparam, sigma] = est_GGD_param(structdis_diff_1(:));
ftrs = [ftrs gamparam sigma  ];
[gamparam, sigma] = est_GGD_param(structdis_diff_2(:));
ftrs = [ftrs gamparam sigma  ];
end


%% ================================================================
%
% estimateggdparam() computes generalized Gaussian distribution parameters
% from provided samples
%
%   This code makes use of the 'BRISQUE Software Release' implementation by Anish Mittal. 
%   (http://live.ece.utexas.edu/research/quality/BRISQUE_release.zip)
%
%============================================================================
function [beta_par, alpha_par] = est_GGD_param(vec)
% moment matching to estimate \beta
gam                              = 0.1:0.001:6;
r_gam                            = (gamma(1./gam).*gamma(3./gam))./((gamma(2./gam)).^2);
sigma_sq                         = mean((vec).^2);
alpha_par                            = sqrt(sigma_sq);
E                                = mean(abs(vec));
rho                              = sigma_sq/E^2;
[~, array_position] = min(abs(rho - r_gam));
beta_par                         = gam(array_position);
% close form equation of entropy
% entr_par = (1/beta_par) - log(beta_par/(2*alpha_par*gamma(1/beta_par)));
% close form equation of kurtosis
% kurt_par = gamma(5/beta_par)*gamma(1/beta_par)/(gamma(3/beta_par)^2);
end

%% ================================================================
%
% estimateaggdparam() computes the assymetric generalized Gaussian distribution 
% parameters from provided samples
%
%   This code makes use of the 'BRISQUE Software Release' implementation by Anish Mittal. 
%   (http://live.ece.utexas.edu/research/quality/BRISQUE_release.zip)
%
%============================================================================
function [alpha, leftstd, rightstd] = est_AGGD_param(vec)
    gam   = 0.1:0.001:6;
    r_gam = ((gamma(2./gam)).^2)./(gamma(1./gam).*gamma(3./gam));

    leftstd            = sqrt(mean((vec(vec<0)).^2));
    rightstd           = sqrt(mean((vec(vec>0)).^2));
    gammahat           = leftstd/rightstd;
    rhat               = (mean(abs(vec)))^2/mean((vec).^2);
    rhatnorm           = (rhat*(gammahat^3 +1)*(gammahat+1))/((gammahat^2 +1)^2);
    [~, array_position] = min((r_gam - rhatnorm).^2);
    alpha              = gam(array_position);
end



function param = nakafit(data)
param(1) = mean(data(:));
param(2) = (mean(data(:))./std(data(:))).^2;
end
