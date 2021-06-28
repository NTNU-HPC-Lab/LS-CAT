#include "includes.h"
__global__ void add(int *a, int *b, int *c, int n){
int index = threadIdx.x + blockIdx.x * blockDim.x;
if(index < n)
c[index] = a[index] + b[index];
}