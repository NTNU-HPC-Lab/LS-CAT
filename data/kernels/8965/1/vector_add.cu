#include "includes.h"
__global__ void vector_add(int *a, int *b, int *c){
int index =  blockIdx.x * blockDim.x + threadIdx.x;
c[index] = a[index] + b[index] ;
}