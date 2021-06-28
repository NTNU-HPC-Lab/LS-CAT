#include "includes.h"
__global__ void init_cs(int *d_cl, int *d_cs, int c_size, int chunk)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= c_size) {
return;
}

if (i == 0) {
d_cs[i] = 0;
}
else {
d_cs[i] = d_cl[i - 1] * chunk;
}

}