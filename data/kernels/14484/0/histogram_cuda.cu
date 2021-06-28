#include "includes.h"
__global__ void	histogram_cuda(int *histogram, float *values, size_t nb, float bin_size, float min, int bins, int nb_thread)
{
// nb = total size of elems


int			id = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS_PER_THREAD;
int			thread_id = threadIdx.x;
int			*local_hist = (int *)malloc(sizeof(int) * bins);

if (id == 0)
printf("Bin size : %f\n", bin_size);

// Init local histogram
for (int i = 0; i < bins; i++)
local_hist[i] = 0;

// One shared array per bin
extern __shared__ int s_bins[];

// Compute serially local bin
for (int i = 0; i < ITEMS_PER_THREAD; i++)
{
for (int j = 0; j < bins; j += 1)
{
// if (id + i < NB)
// 	printf("values[%d] = %f <= %f\n", id + i, values[id + i], (float)min + (float)(j + 1) * bin_size);

if (id + i < nb && values[id + i] <= ((float)min + (float)(j + 1) * bin_size))
{
local_hist[j] += 1;
//printf("BlockIdx : %d - Thread %d : values[%d] = %f -> local_hist[%d] = %d\n", blockIdx.x, thread_id, id + i, values[id + i], j, local_hist[j]);
break ;
}
}
}
__syncthreads();
// Store local bins into shared bins
for (int i = 0; i < bins; i++)
{
s_bins[THREADS * i + thread_id] = local_hist[i];
//		printf("Block %d - Thread %d : s_bins[%d] = local_hist[%d] = %d\n", blockIdx.x, thread_id, THREADS * i + thread_id, i, local_hist[i]);
}

__syncthreads();

// if (thread_id == 0)
// {
// 	for (int i = 0; i < nb_thread * 3; i++)
// 	{
// 		printf("s_bins[%d] = %d\n", i, s_bins[i]);
// 	}
// }

// Reduce each shared bin
// int size = (blockIdx.x == gridDim.x - 1) ? (NB % blockDim.x) : blockDim.x;

int size = THREADS;

for (size_t s = THREADS / 2; s > 0; s >>= 1)
{
if (thread_id + s < THREADS && thread_id < s)
{
for (size_t j = 0; j < bins; j++)
{
s_bins[j * THREADS + thread_id] = s_bins[j * THREADS + thread_id] + s_bins[j * THREADS + thread_id + s];

if (size % 2 == 1 && thread_id + s + s == size - 1)
s_bins[j * THREADS + thread_id] = s_bins[j * THREADS + thread_id] + s_bins[j * THREADS + thread_id + s + s];
}
}
__syncthreads();
size = s;
}

// Store the result into histogram
if (thread_id == 0)
{
for (int i = 0; i < bins; i++) {
histogram[i + blockIdx.x * bins] = s_bins[THREADS * i];




//		histogram[0 + blockIdx.x * bins] = s_bins[0];
//		histogram[1 + blockIdx.x * bins] = s_bins[THREADS];
//		histogram[2 + blockIdx.x * bins] = s_bins[THREADS * 2];
//		printf("histogram[%d] = %d\n", 0 + blockIdx.x * bins, s_bins[0]);
//		printf("histogram[%d] = %d\n", 1 + blockIdx.x * bins, s_bins[THREADS]);
//  printf("histogram[%d] = %d\n", i + blockIdx.x * bins, s_bins[THREADS * i]);
}
}
}