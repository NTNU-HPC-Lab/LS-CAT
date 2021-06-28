#include "includes.h"


#define threads 32
#define size 5

using namespace std;




__global__ void callOperation(int *a, int *b, int *res, int k, int p, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n) {
return;
}

int tid = tidx * n + tidy;

res[tid] = a[tid] - b[tid];
if (res[tid] < k) {
res[tid] = p;
}
}