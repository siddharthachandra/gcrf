function res = denseCRF(im, unary, nInferenceIterations, newOpts)
% DENSECRF Wrapper for Philipp Krähenbühl's Fully Connected CRF code [1].
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
%       Philipp Krähenbühl and Vladlen Koltun, NIPS 2011
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
opts.xstdUnary  = 3;
opts.ystdUnary  = 3;
opts.wUnary     = 3;
opts.xstdBinary = 69;
opts.ystdBinary = 69;
opts.wBinary    = 5;
opts.rgbStd     = 5;
opts.logprob    = true; %
if nargin == 4, opts = updateOpts(opts,newOpts); end
if opts.logprob, unary = logprob(unary); end % turn unaries to log-probabilities
% Reshape and permute input image and unary scores for mex file.
% Matlab processes arrays in column-major order, so we have to permute the 
% height and width dimensions before reshaping.
% The unary scores must have dimensions nLabels x (H*W). 
unary = reshape(permute(unary, [3 2 1]), [nLabels, imWidth*imHeight]);
im    = permute(im, [3 2 1]);

% Run inference
res = denseInferenceMex(im(:), unary, opts);

% Reshape and permute again
res = permute(reshape(res, nLabels, imWidth, imHeight), [3 2 1]);

function [u, p] = logprob(scores)
% Turn the cnn coarse scores to log probabilities
exps = exp(bsxfun(@minus,scores,max(scores,[],3))); % subtract max to avoid overflow
p    = bsxfun(@rdivide, exps, sum(exps,3)); % softmax probabilities
u    = -log(p);                             % unary

function opts = updateOpts(opts,newOpts)
% Update opts fields using input struct
fnames = fieldnames(newOpts);
for i=1:numel(fnames)
    opts.(fnames{i}) = newOpts.(fnames{i});
end
