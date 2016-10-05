/*
    Copyright (c) 2013, Philipp Kr??henb??hl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Kr??henb??hl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Kr??henb??hl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "densecrf.h"
#include <cstdio>
#include <cmath>
//#include "ppm.h"
//#include "common.h"
#include "/usr/local/MATLAB/R2013a/extern/include/mex.h"
#include "/usr/local/MATLAB/R2013a/extern/include/matrix.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
   // mexPrintf("1");
    // Input arguments
    const unsigned char *im = static_cast<const unsigned char *>(mxGetData(prhs[0]));
    const float *pUnary     = static_cast<const float *>(mxGetData(prhs[1]));
    const mxArray *mxOpts   = prhs[2];
    //mexPrintf("2");
    // Image and annotation dimensions
    const int W  = static_cast<const int>(mxGetScalar(mxGetField(mxOpts, 0, "imWidth")));
    const int H  = static_cast<const int>(mxGetScalar(mxGetField(mxOpts, 0, "imHeight")));
    const int M  = static_cast<const int>(mxGetScalar(mxGetField(mxOpts, 0, "nLabels")));
    const int nIter = static_cast<const int>(mxGetScalar(mxGetField(mxOpts, 0, "nIter")));
    //mexPrintf("3");
    // Unary and pairwise parameters
    const float xstdUnary  = static_cast<const float>(mxGetScalar(mxGetField(mxOpts, 0, "xstdUnary")));
    const float ystdUnary  = static_cast<const float>(mxGetScalar(mxGetField(mxOpts, 0, "ystdUnary")));
    const float wUnary     = static_cast<const float>(mxGetScalar(mxGetField(mxOpts, 0, "wUnary")));
    const float xstdBinary = static_cast<const float>(mxGetScalar(mxGetField(mxOpts, 0, "xstdBinary")));
    const float ystdBinary = static_cast<const float>(mxGetScalar(mxGetField(mxOpts, 0, "ystdBinary")));
    const float wBinary    = static_cast<const float>(mxGetScalar(mxGetField(mxOpts, 0, "wBinary")));
    const float rgbStd     = static_cast<const float>(mxGetScalar(mxGetField(mxOpts, 0, "rgbStd")));
//mexPrintf("4");
    // Convert input unary scores to an Eigen::Matrix
    Map<const MatrixXf> unary(pUnary, M, W*H);
//mexPrintf("a");
    // Setup the CRF model
    DenseCRF2D crf(W, H, M);
//mexPrintf("b");
    // Specify the unary potential as an array of size W*H*(#classes)
    // packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
    crf.setUnaryEnergy( unary );
//mexPrintf("v");
    // add a color independent term (feature = pixel location 0..W-1, 0..H-1)
    crf.addPairwiseGaussian(xstdUnary, ystdUnary, new PottsCompatibility(wUnary));
    // add a color dependent term (feature = xyrgb)
    //mexPrintf("d");
    crf.addPairwiseBilateral( xstdBinary, ystdBinary, rgbStd, rgbStd, rgbStd,
                              im, new PottsCompatibility( wBinary ) );

    //mexPrintf("z");
    // Do map inference  Q: nLabels x (H*W)
    MatrixXf Q = crf.startInference(), t1, t2;
      //  mexPrintf("g");

    for( int it=0; it<nIter; it++ ) crf.stepInference( Q, t1, t2 );
   // mexPrintf("r");


    //  Return output as inference scores
    mwSize outDims[2]; outDims[0] = M; outDims[1] = H*W;
    plhs[0] = mxCreateNumericArray(2,outDims,mxSINGLE_CLASS,mxREAL);
    float *out = static_cast<float *>(mxGetData(plhs[0]));
    Map<MatrixXf>(out, M, H*W) = Q;

}
