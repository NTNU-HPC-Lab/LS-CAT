#include "includes.h"


#define N 512
__global__ void calculate(int *a, int *b, int *c){
c[threadIdx.x] = ((a[threadIdx.x]+2)+b[threadIdx.x])*3;
}