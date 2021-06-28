#include "includes.h"
__global__ void  check_support(float * vec_input, float * vec, const int n, int * support_counter)
{
int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

if (xIndex < n) {
if ( vec_input[xIndex] != 0 ) {
if (vec[xIndex] != 0) {
atomicAdd(support_counter, 1);
}
}
else {
if (vec[xIndex] == 0) {
atomicAdd(support_counter + 1, 1);
}
}
}
}