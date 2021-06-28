#include "includes.h"
__global__ void knapsackGPU(int* dp, int row, int* d_value, int* d_weight,int capacity)
{
int in = threadIdx.x + (blockDim.x * blockIdx.x);
if (row != 0)
{
int ind = in + (row * (capacity+1));
if (in <= (capacity+1) && in > 0)
{
if (in >= d_weight[row - 1])
{
dp[ind] = dp[ind - (capacity+1)]> (d_value[row - 1] + dp[ind - (capacity + 1) - d_weight[row - 1]]) ? dp[ind - (capacity + 1)] : (d_value[row - 1] + dp[ind - (capacity + 1) - d_weight[row - 1]]);
}
else
dp[ind] = dp[ind - (capacity+1)];
}
if (in == 0)
{
dp[ind] = 0;
}
}
else
{
dp[in] = 0;
}
}