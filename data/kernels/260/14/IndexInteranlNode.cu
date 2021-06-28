#include "includes.h"
__global__ void IndexInteranlNode(bool *forest, int base, int step)
{
int left  = 2*(base+threadIdx.x);
int right = left + 1;
int offset = blockIdx.x*step;
forest[offset+base+threadIdx.x] = (forest[offset+left]&&forest[offset+right]);
}