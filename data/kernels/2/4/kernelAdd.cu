#include "includes.h"



// Possible weight coefficients for tracking cost evaluation :
// Gaussian discretisation
/*
*       1  4  6  4  1
*       4 16 24 16  4
*       6 24 36 24  6
*       4 16 24 16  4
*       1  4  6  4  1
*/


// Compute spatial derivatives using Scharr operator - Naive implementation..

// Compute spatial derivatives using Scharr operator - Naive implementation..

// Compute spatial derivatives using Sobel operator - Naive implementation..

// Compute spatial derivatives using Sobel operator - Naive implementation..



// Low pass gaussian-like filtering before subsampling

// Low pass gaussian-like filtering before subsampling

/*
// Upsample a picture using the "magic" kernel
*/
__global__ void kernelAdd(float const *in1, float const *in2, int _w, int _h, float *out) {

int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

if(x >= _w || y >= _h)
return;

out[y*_w + x] = __fsqrt_rn(__fadd_rn(__fmul_rn(in1[y*_w + x],in1[y*_w + x]), __fmul_rn(in2[y*_w + x],in2[y*_w + x])));
}