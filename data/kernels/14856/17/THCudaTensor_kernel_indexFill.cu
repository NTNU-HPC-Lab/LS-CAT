#include "includes.h"
__global__ void THCudaTensor_kernel_indexFill( float *tensor, long* stride, float *index, long src_nDim, int dim, long idx_size, long tensor_size, long size_dim, float val )
{
int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

long flat_size = tensor_size / idx_size;

if (thread_idx < flat_size)
{
long coeff = 0;
for (int i=0; i<idx_size; i++)
{
int leftover = thread_idx;
int srcIdx = 0;
for (int d=0; d<src_nDim; d++)
{
if (d < dim)
{
coeff = leftover / (stride[d] / size_dim);
leftover -= coeff * (stride[d] / size_dim);
srcIdx += coeff * stride[d];
}
else if (d > dim)
{
coeff = leftover / stride[d];
leftover -= coeff * stride[d];
srcIdx += coeff * stride[d];
}
}
tensor[srcIdx + (long)((index[i])-1)*stride[dim]] = val;
}
}
}