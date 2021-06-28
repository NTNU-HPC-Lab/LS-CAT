#include "includes.h"



#define SIZE 16


__global__ void compare(int *in_d, int* out_d)
{
if (in_d[blockIdx.x] == 6)
{
out_d[blockIdx.x] = 1;
}
else
out_d[blockIdx.x] = 0;
}