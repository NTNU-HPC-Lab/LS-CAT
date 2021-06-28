#include "includes.h"
__global__ void compare_2_arrays_kernel(float *one, float *two, int size)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index >= size) return;

const float diff = 100 * fabs(one[index] - two[index]) / fabs(one[index]);

if (diff > 10) printf(" i: %d - one = %f, two = %f, diff = %f %% \n", index, one[index], two[index], diff);
}