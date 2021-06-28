#include "includes.h"
__global__ void Kernel(int* a,int* b,int *c,int n){

int i = blockIdx.x*blockDim.x + threadIdx.x;

__shared__ extern int shared_mem[];
int reg;

if(i>= n) return;

reg = a[i] + b[i];
shared_mem[i] = reg;
c[i] = shared_mem[i];

}