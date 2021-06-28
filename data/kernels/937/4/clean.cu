#include "includes.h"
__global__ void clean(unsigned int * e, int n)
{
e[threadIdx.x % n] = 0;
}