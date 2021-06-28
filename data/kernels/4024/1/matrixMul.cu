#include "includes.h"
__global__ void matrixMul(int *a, int *b, int *c){
int my_x, my_y;
my_x = blockIdx.x*blockDim.x + threadIdx.x;
my_y = blockIdx.y*blockDim.y + threadIdx.y;
int local_c = 0;
for(int i = 0 ; i < size; i++)
local_c += a[my_x * size + i] * b[i * size + my_y];

c[my_x * size + my_y ] = local_c;

}