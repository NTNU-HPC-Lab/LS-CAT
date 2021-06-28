#include "includes.h"
__global__ void print_long(long* x, int leng) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < leng) {
printf("%ld,", x[ i ]);
}
}