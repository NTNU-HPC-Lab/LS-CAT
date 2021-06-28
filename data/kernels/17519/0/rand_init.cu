#include "includes.h"
#define FALSE 0
#define TRUE  1
// returns random integer from 1 to lim
__global__ void rand_init(long *a,long seed)
{
int tid=threadIdx.x+blockDim.x*blockIdx.x;
//long a = 100001;
a[tid] = seed + tid;
}