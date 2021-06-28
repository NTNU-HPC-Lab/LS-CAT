#include "includes.h"
__global__ void update_density_rg_1_real_gpu_kernel(int size__, double const* psi_rg__, double wt__, double* density_rg__)
{
int ir = blockIdx.x * blockDim.x + threadIdx.x;
if (ir < size__) {
double p = psi_rg__[ir];
density_rg__[ir] += p * p * wt__;
}
}