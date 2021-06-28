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


__device__ float dp_criteria(float *dp, int ind, int depth_dim_size, int d, float P_one, float P_two, float * d_zero, float * d_one, float * d_two, float * d_three){
*d_zero = dp[ind];
if (d > 0)
*d_one = dp[ind - depth_dim_size] + P_one;
else
*d_one = 10000000;

if (d < D-1)
*d_two = dp[ind + depth_dim_size] + P_one;
else
*d_two = 10000000;
return fminf(fminf(*d_zero, *d_one), fminf(*d_two, *d_three)) - *d_three + P_two;

}
__global__ void __diagonal_tl_br_aggregate(float *dp, float *cost_image, int m, int n)
{
// which column of array to work on
int start_col = blockDim.x * blockIdx.x + threadIdx.x + 1;
int depth_dim_size = m*n;

// todo: maybe it will work better to take running average of every d
// slices
while(start_col < n)
{
int col = start_col;
for (int row = 1; row < m; row++)
{
//int arr_ind = 0;
float prev_min = 100000000.0;
int ind = (row - 1) * n + col - 1;

// calculate min cost disparity for this column from row-1
//#pragma unroll
for (int depth = 0; depth < D; depth+=D_STEP){
prev_min = fminf(dp[ind], prev_min);
ind += (depth_dim_size * D_STEP);
}

float d0 = 0;
float d1 = 0;
float d2 = 0;
float d3 = prev_min + (float) P2;
ind = (row - 1) * n + col - 1;
int current_ind = row * n + col;


// todo: try having this loop go from 1 to d-1 and removing the if else
for (int d = 0; d < D; d+=D_STEP){
// for each d I need dp[{d-1, d, d+1}, row-1, col],
dp[current_ind] += cost_image[current_ind] + dp_criteria(dp, ind, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
ind += (depth_dim_size * D_STEP);
current_ind += (depth_dim_size * D_STEP);
}

col += 1;
if (col == n) // wrap each thread around once it gets to the last column
col = 1;

}
start_col += blockDim.x;
}
}