#include "includes.h"
__global__ void addByThreads(int *a, int *b, int *c)
{
// a block can be split into parallel threads.
c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}