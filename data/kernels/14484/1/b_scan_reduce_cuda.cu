#include "includes.h"
__global__ void	b_scan_reduce_cuda(int *values, int *cumulative, size_t bins)
{

int		id = blockDim.x * blockIdx.x + threadIdx.x;
int		tid = threadIdx.x;

int		nb_here = (bins - (blockDim.x * blockIdx.x) < blockDim.x) ? bins - (blockDim.x * blockIdx.x) : blockDim.x;

if (tid == 0)
printf("Block %d nb_here = %d\n", blockIdx.x, nb_here);

////////// REDUCE
//
//
int		next_th = 2;
// printf("size = %u\n", size);
for (int threshold = 1; threshold < blockDim.x / 2; threshold = threshold << 1)
{
// printf("Thread %d \t id = %d \t threshold = %d \t NB = 8 \t threadIdx.x = %d \t next_th = %d\n", tid, id, threshold, tid, next_th);
// printf("NB %d \t tid = %d \t next_th = %d\n", (NB - 1), tid, next_th);
if (tid < nb_here && tid - threshold >= 0 && ((nb_here - 1) - tid) % next_th == 0)
{
//	printf("HERE : values[%d] = %d\n", id, values[id] + values[id - threshold]);
values[id] = values[id] + values[id - threshold];
}
next_th = next_th << 1;
__syncthreads();
}

////////// DOWNSWEEP
//
//

values[nb_here - 1] = 0;
next_th = nb_here >> 1;
int	tmp;
for (int threshold = nb_here; threshold > 1; threshold >>= 1)
{
if (tid == 0)
printf("threshold = %d && next_th = %d\n", threshold, next_th);
// printf("threshold = %d\n", threshold);
if (tid < nb_here && tid - next_th >= 0 && (nb_here - 1 - tid) % threshold == 0) {
tmp = values[id];
printf("Thread %d : values[%d] = %d\n", tid, id, tmp + values[id - threshold]);
printf("Thread %d : values[%d] = %d\n", tid, id - threshold, tmp);
values[id] += values[id - next_th];
values[id - next_th] = tmp;
}
next_th = next_th >> 1;
if (tid == 0)
printf("\n");
__syncthreads();
}


// Store into cumulative
cumulative[id] = values[id];
}