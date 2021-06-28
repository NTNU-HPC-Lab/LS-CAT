#include "includes.h"


#define N 2560
#define M 512
#define BLOCK_SIZE (N/M)
#define RADIUS 5

__global__ void add(double *a, double *b, double *c, int n){
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if(idx < n){
c[idx] = a[idx] + b[idx];
}
}