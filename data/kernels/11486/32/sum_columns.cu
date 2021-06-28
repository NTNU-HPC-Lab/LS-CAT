#include "includes.h"
__global__ void sum_columns ( float * w_mtx, float * output, unsigned int height, unsigned int width )
{
// X thread iterates Columns and sums their Row values
int x = blockIdx.x * blockDim.x + threadIdx.x;
float total;
for ( int y = 0; y < height; y++ )
{
total = __fadd_rz( total, w_mtx[y*width+x]);
}
output[x] = total;
}