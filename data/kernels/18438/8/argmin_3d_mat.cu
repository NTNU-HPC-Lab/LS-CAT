#include "includes.h"
//#define DEPTH 2



// dp - cost aggregation array
// cost_image - m x n x D array
// d - use every d channels of input to conserve register memory
// m - image rows
// n - image columns
// D - depth
// depth_stride - pitch along depth dimension
// row_stride - pitch along row dimension


__global__ void argmin_3d_mat(float * dp, int * stereo_im, int m, int n)
{
int col = blockDim.x * blockIdx.x + threadIdx.x;
int imsize = m*n;
int loop_limit = D*m*n;

while(col < n)
{
int row = blockDim.y * blockIdx.y + threadIdx.y;
while(row < m)
{
int min_ind = -1;
float current_min = 100000000.0;
int current_val = row * n + col;
int v = 0;

for (int depth = 0; depth < loop_limit; depth+=imsize){

if (dp[depth + current_val] < current_min)
{
min_ind = v;
current_min = dp[depth + current_val];
}
v++;
}
stereo_im[current_val] = min_ind;
row+=blockDim.y;
}
col+=blockDim.x;
}
}