#include "includes.h"



/*
* Description:
*/

__device__ int translate_idx(int ii, int d1, int d2, int d3, int d4, int scale_factor_t, int scale_factor_y, int scale_factor_x)
{
int x, y, t, z, w;


w = ii % d4;
ii = ii/d4;
z = ii % d3;
ii = ii/d3;
t = ii % d2;
ii = ii/d2;
y = ii % d1;
ii = ii/d1;
x = ii;
w = w/scale_factor_x;
z = z/scale_factor_y;
t = t/scale_factor_t;
d2 /= scale_factor_t;
d3 /= scale_factor_y;
d4 /= scale_factor_x;
return (((((x*d1+y)*d2)+t)*d3)+z)*d4+w;

}
__global__ void upscale(float *input, float *output, long no_elements, int scale_factor_t, int scale_factor_y, int scale_factor_x, int d1, int d2, int d3, int d4)
{
// output offset:
long ii = threadIdx.x + blockDim.x * blockIdx.x;
ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
if (ii >= no_elements) return;
int ipidx = translate_idx(ii, d1, d2, d3, d4, scale_factor_t, scale_factor_y, scale_factor_x);
output[ii]=input[ipidx];
}