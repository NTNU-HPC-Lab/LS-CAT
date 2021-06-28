#include "includes.h"
__global__ void add(int *c , int *d){
int tid=threadIdx.x;
d[tid]+=c[tid];
}