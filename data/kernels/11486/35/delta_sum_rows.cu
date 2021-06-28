#include "includes.h"
__global__ void delta_sum_rows ( float * w_ik_d, float * delta_i, unsigned int width )
{
// X thread iterates Rows and Sums the respective Column values
int x = blockIdx.x * blockDim.x + threadIdx.x;
float total = 0.f;
for ( int y = 0; y < width; y++ )
{
//printf("X:%d, Σ: %.9f + %.9f\n",x,total,w_ik_d[x*width+y]);
total = __fadd_rz( total, w_ik_d[x*width+y]);
}
delta_i[x] = total;
}