#include "includes.h"

/* https://zxi.mytechroad.com/blog/dynamic-programming/leetcode-730-count-different-palindromic-subsequences/ */


long kMod = 1000000007;

__global__ void helperKernel(char *S, int *dp, int n, long kMod, int len) {
for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n - len; i += blockDim.x * gridDim.x) {
int j = i + len; // jth element is the end of current string
if(S[i] == S[j]) { // if front and rear are the same
dp[i * n + j] = dp[(i + 1) * n + (j - 1)] * 2;
int left = i + 1;
int right = j - 1;

while(left <= right && S[left] != S[i]) {
left++;
}
while(left <= right && S[right] != S[i]) {
right--;
}

if(left == right) {
dp[i * n + j] += 1;
} else if(left > right) {
dp[i * n + j] += 2;
} else {
dp[i * n + j] -= dp[(left + 1) * n + (right - 1)];
}
} else {
dp[i * n + j] = dp[i * n + (j - 1)] + dp[((i + 1) * n) + j] - dp[(i + 1) * n + (j - 1)];
}

dp[i * n + j] = (dp[i * n + j] + kMod) % kMod; // perform positive modulo
}
//__syncthreads();
}