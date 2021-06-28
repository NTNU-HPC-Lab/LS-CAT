#include "includes.h"

#define DOUBLE

#ifdef DOUBLE
#define Complex  cufftDoubleComplex
#define Real double
#define Transform CUFFT_Z2Z
#define TransformExec cufftExecZ2Z
#else
#define Complex  cufftComplex
#define Real float
#define Transform CUFFT_C2C
#define TransformExec cufftExecC2C
#endif

#define TILE_DIM  8

// synchronize blocks
__global__ void spread(Real* src, unsigned int spitch, Real* dst, unsigned int dpitch)
{
unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
unsigned int tid = threadIdx.x;

Real res = (tid >= spitch) ? src[bid * spitch + tid-spitch] : 0.0;
if( tid < dpitch) {
dst[bid * dpitch + tid] = res;
}
}