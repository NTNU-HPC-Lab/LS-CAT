#include "includes.h"
__global__ void add_block(int *a, int *b, int *c){
c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}