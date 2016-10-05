function res = denseCRF(im, unary, nInferenceIterations, newOpts,gvf,stds)
% DENSECRF Wrapper for Philipp Kr??henb??hl's Fully Connected CRF code [1].
%   This function assumes that the input image is a HxWx3 RGB image and
%   that the unary terms are a HxWxK array, where K is the number of
%   classes.
% 
%   res = DENSECRF(im, unary), where im is a HxW uint8 image and unary is a
%   HxWxN array of unary score maps (one per each label class). 
% 
%   res = DENSECRF(im, unary, nInferenceIterations) runs inference for
%   nInferenceIterations (default: 10).
% 
%   res = DENSECRF(im, unary, nInferenceIterations, newOpts) uses input
%   struct newOpts to update default parameter values (copies values from
%   respective fields).
% 
% [1]: Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
%       Philipp Kr??henb??hl and Vladlen Koltun, NIPS 2011
% 
% Stavros Tsogkas, <stavros.tsogkas@centralesupelec.fr>
% Last update: March 2015 

if nargin < 3, nInferenceIterations = 10; end

assert(isa(im,'uint8'),     'Input image must be uint8');
assert(isa(unary,'single'), 'Unary term must be float');
[imHeight, imWidth, ~] = size(im);
[unaryHeight, unaryWidth, nLabels] = size(unary);
assert(imHeight == unaryHeight && imWidth == unaryWidth, ...
    'Image and unary score dimensions do not match')


% Set dedault values for parameters
opts.nIter      = nInferenceIterations;
opts.imHeight   = imHeight;
opts.imWidth    = imWidth;
opts.nLabels    = nLabels;
%opts.wSmooth     = pos_w;
%opts.xstdSmooth  = pos_x_std;
%opts.ystdSmooth  = pos_x_std;

%opts.wBilateral     = bi_w;
%opts.xstdBilateral  = bi_x_std;
%opts.ystdBilateral  = bi_x_std;
%opts.rgbStd         = bi_r_std;

opts.wUnary     = 3;  %('wSmooth')
opts.xstdUnary  = 3;  %('xstdSmooth')
opts.ystdUnary  = 3;  %('ystdSmooth')

opts.xstdBinary = 50;
opts.ystdBinary = 50;
opts.wBinary    = 5;

opts.rgbStd     = 10;
opts.logprob    = true; %
opts.useGvf     = 0;
opts.gvfStd     = .2;

if nargin >= 4, opts = updateOpts(opts,newOpts); end

opts.wUnary     = opts.wSmooth;
opts.xstdUnary = opts.xstdSmooth;
opts.ystdUnary = opts.ystdSmooth;

opts.wBinary = opts.wBilateral;
opts.xstdBinary = opts.xstdBilateral;
opts.ystdBinary = opts.ystdBilateral;

%if opts.logprob, unary = logprob(unary); end % turn unaries to log-probabilities
%opts

% Reshape and permute input image and unary scores for mex file.
% Matlab processes arrays in column-major order, so we have to permute the 
% height and width dimensions before reshaping.
% The unary scores must have dimensions nLabels x (H*W). 
unary = reshape(permute(unary, [3 2 1]), [nLabels, imWidth*imHeight]);
if (opts.useGvf)&(nargin>=5)
    im = cat(3,im,gvf);
    opts.nDims = size(gvf,3);
    opts.stds = single(stds);
else
    opts.useGvf = 0;
end
im    = permute(im, [3 2 1]);

% Run inference
if opts.useGvf
    res = denseInferenceMexEdge(im(:), unary, opts,opts.stds);
else
    res = denseInferenceMex(im(:), unary, opts);
end
% Reshape and permute again
res = permute(reshape(res, nLabels, imWidth, imHeight), [3 2 1]);

function [u, p] = logprob(scores)
% Turn the cnn coarse scores to log probabilities
exps = exp(scores-max(scores(:)));          % subtract max to avoid overflow
p    = bsxfun(@rdivide, exps, sum(exps,3)); % softmax probabilities
u    = -log(p);                             % unary

function opts = updateOpts(opts,newOpts)

% Update opts fields using input struct
fnames = fieldnames(newOpts);
for i=1:numel(fnames)
    opts.(fnames{i}) = newOpts.(fnames{i});
end
