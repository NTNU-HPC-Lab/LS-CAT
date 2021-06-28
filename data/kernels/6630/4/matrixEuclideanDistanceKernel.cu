#include "includes.h"




using namespace std;

__global__ void matrixEuclideanDistanceKernel(float* in, float* out, int n, int m){
extern __shared__ float Rs[];
float tmp, s;
int myRow = blockIdx.x*blockDim.x + threadIdx.x;
for (int r = 0; r<n; r++){ //outer loop
s = 0;
for (int i = 0; i <= m / 256; i++){
if (i * 256 + threadIdx.x < m)
Rs[i * 256 + threadIdx.x] = in[r*m + i * 256 + threadIdx.x];
}
__syncthreads();
for (int i = 0; i<m && myRow<n; i++){
tmp = Rs[i] - in[myRow*m + i];
s += tmp*tmp;
}
if (myRow<n)
out[myRow*n + r] = s;
__syncthreads();
}
}