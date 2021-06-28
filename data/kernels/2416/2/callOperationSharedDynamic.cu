#include "includes.h"


#define threads 32
#define size 5

using namespace std;




__global__ void callOperationSharedDynamic(int *a, int *b, int *res, int k, int p, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n) {
return;
}

int tid = tidx * n + tidy;

extern __shared__ int data[];

int *s_a = data;
int *s_b = &s_a[size * size];
int *s_res = &s_b[size * size];

__shared__ int s_p, s_k;

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