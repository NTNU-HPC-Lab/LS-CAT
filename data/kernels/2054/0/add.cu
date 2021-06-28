#include "includes.h"
#define N 512

__global__ void add(int *a, int *b, int *c){
c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}