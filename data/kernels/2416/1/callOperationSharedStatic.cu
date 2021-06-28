#include "includes.h"


#define threads 32
#define size 5

using namespace std;




__global__ void callOperationSharedStatic(int *a, int *b, int *res, int k, int p, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n) {
return;
}

int tid = tidx * n + tidy;

__shared__ int s_a[size * size], s_b[size * size], s_res[size * size], s_p, s_k;

s_k = k;
s_p = p;
s_a[tid] = a[tid];
s_b[tid] = b[tid];

s_res[tid] = s_a[tid] - s_b[tid];
if (s_res[tid] < s_k) {
s_res[tid] = s_p;
}

res[tid] = s_res[tid];
}