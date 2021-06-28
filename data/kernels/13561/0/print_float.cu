#include "includes.h"
__global__ void print_float(float* x, int leng) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < leng) {
printf("%f,", x[ i ]);
}
}