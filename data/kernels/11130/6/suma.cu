#include "includes.h"
__global__ void suma( int *a, int *b, int *c, int n, int m) {
int index = blockIdx.x + blockIdx.y * blockDim.y;
if(index < n*m){
c[index] = a[index] + b[index];
}
}