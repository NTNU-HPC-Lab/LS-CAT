#include "includes.h"
__global__ void print_double(double* x, int leng) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < leng) {
printf("%lf,", x[ i ]);
}
}