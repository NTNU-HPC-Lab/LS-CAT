#include "includes.h"
__global__ void THCudaTensor_kernel_renorm(float *data, const float value, const long size, const float maxnorm)
{
__shared__ float buffer[32];
long tx = threadIdx.x;
long bx = blockIdx.x;
long step = blockDim.x;
float *row = data + size*bx;

buffer[tx] = 0;

// get norm of axis
for (long i=tx; i<size; i+=step)
{
buffer[tx] += pow(fabs(row[i]), value);
}
// add (reduce)
for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
{
__syncthreads();
if (tx < stride)
buffer[tx] += buffer[tx+stride];
}
// clip norms
__syncthreads();
float norm = pow(buffer[0], 1/value);
if (norm > maxnorm)
{
norm = maxnorm / (norm + 1e-7);
// renormalize
for (long i=tx; i<size; i+=step)
{
row[i] *= norm;
}
}
}