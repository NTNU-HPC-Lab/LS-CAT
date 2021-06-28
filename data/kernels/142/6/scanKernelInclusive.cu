#include "includes.h"
__global__ void scanKernelInclusive(int *c, const int *a, size_t size, size_t offset)
{
int myId =
threadIdx.x;

if (((myId - offset) < size) &&
(myId >= offset))
{
c[myId] = a[myId];

__syncthreads();

size_t _stepsLeft =
size;

unsigned int _neighbor =
1;

while (_stepsLeft)
{
int op1 = c[myId];
int op2 = 0;

if ((myId - offset) >= _neighbor)
{
op2 =
c[myId - _neighbor];
}
else
{
break;
}

__syncthreads();

c[myId] =
op1 + op2;

__syncthreads();

_stepsLeft >>= 1;
_neighbor <<= 1;
}

if (offset > 0)
{
c[myId] +=
c[offset - 1];
}
}
}