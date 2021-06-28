#include "includes.h"
__global__ void set_all_zero_kernel(double *ua_gpu, double *ub_gpu, double *uc_gpu)
{
ua_gpu[blockIdx.x * blockDim.x + blockIdx.y] = 0;
ub_gpu[blockIdx.x * blockDim.x + blockIdx.y] = 0;
uc_gpu[blockIdx.x * blockDim.x + blockIdx.y] = 0;
// TODO: sync CPU after this -> move to utils.cu file
}