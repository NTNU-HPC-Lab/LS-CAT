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
__global__ void spread_y_r(Real* src, Real* dst)
{
unsigned int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
unsigned int tid1 = (blockIdx.y * gridDim.x * 2 + blockIdx.x) * blockDim.x + threadIdx.x;

Real res =  src[tid];
dst[tid1 + blockDim.x*gridDim.x] = res;
#ifdef DOUBLE
dst[tid1] = 0.;
#else
dst[tid1] = 0.f;
#endif
}