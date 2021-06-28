#include "includes.h"
__global__ void print_int(int* x, int leng) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < leng) {
printf("%d,", x[ i ]);
}
}