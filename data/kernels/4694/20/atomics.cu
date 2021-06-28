#include "includes.h"
__global__ void atomics(int *shared_var, int iters)
{
for (int i = 0; i < iters; i++)
{
atomicAdd(shared_var, 1);
}
}