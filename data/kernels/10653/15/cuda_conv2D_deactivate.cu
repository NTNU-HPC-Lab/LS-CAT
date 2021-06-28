#include "includes.h"
__global__ void cuda_conv2D_deactivate(double* err, const double* net, const double* activation, size_t outputs)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if(id >= outputs)
return;
err[id] *= (1.0 - activation[id] * activation[id]);
}