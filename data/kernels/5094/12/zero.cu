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
__global__ void zero(int nx, int ny, int nz, Real *z) {

int tj = threadIdx.x;
int td = blockDim.x;

int blockData = (nx*ny*nz)/(gridDim.x*gridDim.y);

int jj = ((blockIdx.y)*gridDim.x + (blockIdx.x))*blockData;

for (int k=0; k<blockData/td; k++) {
z[jj + tj+ k*td] = 0.0;
}
}