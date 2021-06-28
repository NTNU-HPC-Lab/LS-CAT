#include "includes.h"
__global__ void convert(double* A,double* C)
{
int idx = BLOCK*blockIdx.x + threadIdx.x;
int i;
int stride = BLOCK * THREAD;

for(i=idx;i<SIZE;i+=stride)
A[i] = C[SIZE-i-1];

}