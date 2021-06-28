#include "includes.h"
__global__ void MHDComputedUx_CUDA3_kernel(float *FluxD, float *FluxS1, float *FluxS2, float *FluxS3, float *FluxTau, float *FluxBx, float *FluxBy, float *FluxBz, float *FluxPhi, float *dUD, float *dUS1, float *dUS2, float *dUS3, float *dUTau, float *dUBx, float *dUBy, float *dUBz, float *dUPhi, float dtdx, int size)
{
// get thread and block index
const long tx = threadIdx.x;
const long bx = blockIdx.x;
const long by = blockIdx.y;

int igrid = tx + bx*CUDA_BLOCK_SIZE + by*CUDA_BLOCK_SIZE*CUDA_GRID_SIZE;


if (igrid < 2 || igrid > size - 3)
return;

int igridp1 = igrid + 1;
dUD  [igrid] = (FluxD  [igrid] - FluxD  [igridp1])*dtdx;
dUS1 [igrid] = (FluxS1 [igrid] - FluxS1 [igridp1])*dtdx;
dUS2 [igrid] = (FluxS2 [igrid] - FluxS2 [igridp1])*dtdx;
dUS3 [igrid] = (FluxS3 [igrid] - FluxS3 [igridp1])*dtdx;
dUTau[igrid] = (FluxTau[igrid] - FluxTau[igridp1])*dtdx;
dUBx [igrid] = (FluxBx [igrid] - FluxBx [igridp1])*dtdx;
dUBy [igrid] = (FluxBy [igrid] - FluxBy [igridp1])*dtdx;
dUBz [igrid] = (FluxBz [igrid] - FluxBz [igridp1])*dtdx;
dUPhi[igrid] = (FluxPhi[igrid] - FluxPhi[igridp1])*dtdx;

}