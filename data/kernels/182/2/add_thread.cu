#include "includes.h"
__global__ void add_thread(int *a, int *b, int *c){
c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}