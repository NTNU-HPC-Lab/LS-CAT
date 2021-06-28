#include "includes.h"
__global__ void Caps(char *c, int *b)
{
int tid = blockIdx.x;
if (tid < N)
{
if (b[tid] == 1)
{
int ascii = (int)c[tid];
ascii -= 32;
c[tid] = (char)ascii;
}
}

}