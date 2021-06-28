#include "includes.h"
__global__ void add(int *a, int *b, int *c){

int tid = blockIdx.x;

c[tid] = a[tid] + b[tid];

}