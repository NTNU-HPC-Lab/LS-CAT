#include "includes.h"

/* https://zxi.mytechroad.com/blog/dynamic-programming/leetcode-730-count-different-palindromic-subsequences/ */


long kMod = 1000000007;

__global__ void setData(int *dp, int n) {
for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
dp[i * n + i] = 1;
}
}