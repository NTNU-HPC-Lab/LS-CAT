#include "includes.h"


__global__ void add_kernel(Para* para, int size)
{
int tid = threadIdx.x;

if (tid < size)
{
para->d_c[tid] = para->d_a[tid] + para->d_b[tid];
}
}