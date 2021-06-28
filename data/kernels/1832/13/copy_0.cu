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
__global__ void copy_0(int nx, int ny, int nz, Real *in, Real *out) {

int tj = threadIdx.x;
int td = blockDim.x;

int jj =  (blockIdx.y*nx*ny/4 + blockIdx.x*nx/2);
int jj1 =  ((blockIdx.y+nz/2)*nx*ny + (blockIdx.x+ny/2)*nx);


out[jj1+tj+td] = in[jj+tj];

}