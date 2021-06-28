#include "includes.h"





__global__ void add(int *a, int *b, int *c,int size) {
c[size*blockIdx.x+ threadIdx.x] = a[size*blockIdx.x+ threadIdx.x] + b[size*blockIdx.x+ threadIdx.x];
}