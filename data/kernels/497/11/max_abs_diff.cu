#include "includes.h"
__global__ void max_abs_diff(float* diff, const float* output1, const float* output2, const int size)
{
extern __shared__ float s_max[];
int i = blockDim.x*blockIdx.x + threadIdx.x;
int tx = threadIdx.x;
if (i<size)
{
float o1 = output1[i];
if (o1 == -1)
{
s_max[tx] = -1;
}
else
{
s_max[tx] = fabsf(o1 - output2[i]);
}
}
else
{
s_max[tx] = -1;
}
for (int j = blockDim.x / 2; j> 0; j >>= 1)
{
__syncthreads();
if (tx<j)
{
s_max[tx] = fmaxf(s_max[tx], s_max[tx + j]);
}
}
if (tx == 0)
{
diff[blockIdx.x] = s_max[0];
}
}