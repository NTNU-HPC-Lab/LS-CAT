#include "includes.h"
__global__ void sub3(float *val1, float *val2, int *num_elem)
{
int i = threadIdx.x;
val1[i] += val2[i]+1;
}