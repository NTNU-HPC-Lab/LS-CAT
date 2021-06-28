#include "includes.h"
/*
#define N 512

#define N 2048
#define THREADS_PER_BLOCK 512

*/
const int THREADS_PER_BLOCK = 32;
const int N = 2048;



__global__ void mult(int *a, int *b, int *c)
{
int pos = threadIdx.x + blockDim.x * blockIdx.x;
if (pos >= N) return;

c[pos] = a[pos] * b[pos];
}