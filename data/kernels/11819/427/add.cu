#include "includes.h"
__global__ void add(int *a, int *b, int *c)
{
extern __shared__ int shared_mem[];
int * shmem=shared_mem;
shmem[threadIdx.x]=threadIdx.x;
a[threadIdx.x]=shmem[threadIdx.x];
b[threadIdx.x]=shmem[threadIdx.x];
c[threadIdx.x]=a[threadIdx.x]+b[threadIdx.x];
}