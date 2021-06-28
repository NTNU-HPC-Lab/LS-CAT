#include "includes.h"
__global__ void max_min_cuda(float *d_in1, float *d_in2, float *d_max, float *d_min, size_t nb)
{
int ft_id = threadIdx.x + blockDim.x * blockIdx.x;
int tid = threadIdx.x;
int size = (blockIdx.x == gridDim.x - 1) ? (nb % blockDim.x) : blockDim.x;

for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
{
if (ft_id + s < nb && tid < s)
{
d_in1[ft_id] = (d_in1[ft_id] > d_in1[ft_id + s]) ? d_in1[ft_id] : d_in1[ft_id + s];
if (size % 2 == 1 && ft_id + s + s == size - 1)
d_in1[ft_id] = (d_in1[ft_id] > d_in1[ft_id + s + s]) ? d_in1[ft_id] : d_in1[ft_id + s + s];
d_in2[ft_id] = (d_in2[ft_id] < d_in2[ft_id + s]) ? d_in2[ft_id] : d_in2[ft_id + s];
if (size % 2 == 1 && ft_id + s + s == size - 1)
d_in2[ft_id] = (d_in2[ft_id] < d_in2[ft_id + s + s]) ? d_in2[ft_id] : d_in2[ft_id + s + s];
}
__syncthreads();
size /= 2;
}
if (tid == 0)
{
d_max[blockIdx.x] = d_in1[ft_id];
d_min[blockIdx.x] = d_in2[ft_id];
}
// __syncthreads();
// for (int i = 0; i < GRID_SIZE; i++)
// 	printf("d_out[%d] = %f\n", i, d_out[i]);
}