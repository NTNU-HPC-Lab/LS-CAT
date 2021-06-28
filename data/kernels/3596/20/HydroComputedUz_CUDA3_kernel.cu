#include "includes.h"
__global__ void HydroComputedUz_CUDA3_kernel(float *FluxD, float *FluxS1, float *FluxS2, float *FluxS3, float *FluxTau, float *dUD, float *dUS1, float *dUS2, float *dUS3, float *dUTau, float dtdx, int size, int dim0, int dim1, int dim2)
{
// get thread and block index
const long tx = threadIdx.x;
const long bx = blockIdx.x;
const long by = blockIdx.y;

int igridz = tx + bx*CUDA_BLOCK_SIZE + by*CUDA_BLOCK_SIZE*CUDA_GRID_SIZE;

if (igridz < 2 || igridz > size - 3)
return;

int j = igridz / (dim0*dim2);
int i = (igridz - j*dim0*dim2) / dim2;
int k = igridz - j*dim0*dim2 - i*dim2;
int igrid = i + (j + k*dim1) * dim0;

int igridzp1 = igridz + 1;
j = igridzp1 / (dim0*dim2);
i = (igridzp1 - j*dim0*dim2) / dim2;
k = igridzp1 - j*dim0*dim2 - i*dim2;
int igridp1 = i + (j + k*dim1) * dim0;

dUD  [igrid] += (FluxD  [igrid] - FluxD  [igridp1])*dtdx;
dUS1 [igrid] += (FluxS1 [igrid] - FluxS1 [igridp1])*dtdx;
dUS2 [igrid] += (FluxS2 [igrid] - FluxS2 [igridp1])*dtdx;
dUS3 [igrid] += (FluxS3 [igrid] - FluxS3 [igridp1])*dtdx;
dUTau[igrid] += (FluxTau[igrid] - FluxTau[igridp1])*dtdx;

}