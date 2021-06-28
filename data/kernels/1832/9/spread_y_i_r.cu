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
__global__ void spread_y_i_r(Real* src, Real* dst)
{
unsigned int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
unsigned int tid1 = (blockIdx.y * gridDim.x * 2 + blockIdx.x) * blockDim.x + threadIdx.x;

Real res =  src[tid1];
dst[tid] = res;
}