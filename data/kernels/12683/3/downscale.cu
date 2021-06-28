#include "includes.h"
__device__ int translate_idx_inv(int ii, int d1, int d2, int d3, int d4, int scale_factor_t, int scale_factor_xy, int off_time, int off_x, int off_y)
{
/* d1 = channel
d2 = time
d3, d4 = height, width
*/
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
t = t*scale_factor_t+off_time;
w = w*scale_factor_xy+off_x;
z = z*scale_factor_xy+off_y;
d2 *= scale_factor_t;
d3 *= scale_factor_xy;
d4 *= scale_factor_xy;
return (((((x*d1+y)*d2)+t)*d3)+z)*d4+w;

}
__device__ int translate_idx(int ii, int d1, int d2, int d3, int d4, int scale_factor_t, int scale_factor_xy)
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
w = w/scale_factor_xy;
z = z/scale_factor_xy;
t = t/scale_factor_t;
d2 /= scale_factor_t;
d3 /= scale_factor_xy;
d4 /= scale_factor_xy;
return (((((x*d1+y)*d2)+t)*d3)+z)*d4+w;

}
__global__ void downscale(float *gradInput_data, float *gradOutput_data, long no_elements, int scale_factor_t, int scale_factor_xy, int d1, int d2, int d3, int d4)
{
// output offset:
long ii = threadIdx.x + blockDim.x * blockIdx.x;
ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
if (ii >= no_elements) return;
for (int i=0; i < scale_factor_t; i++){
for(int j=0; j < scale_factor_xy; j++){
for(int k=0; k < scale_factor_xy; k++){
int ipidx = translate_idx_inv(ii, d1, d2, d3, d4, scale_factor_t, scale_factor_xy, i, j, k);
gradInput_data[ii] += gradOutput_data[ipidx];
}
}
}
}