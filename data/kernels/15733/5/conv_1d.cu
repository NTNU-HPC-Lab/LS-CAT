#include "includes.h"
__global__ void conv_1d(int* a, int* c, int n) {
int id = blockIdx.x * blockDim.x + threadIdx.x;

//cal the radius of the mask(mid point)
int r = MASK_LEN / 2;
//cal the start point of for the element
int start = id - r;
int temp = 0;
for (int j = 0; j < MASK_LEN; j++)
{
if ((start + j >= 0) && (start + j < n))
{
temp += a[start + j] * mask[j];
}
}
c[id] = temp;
}