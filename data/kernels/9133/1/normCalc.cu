#include "includes.h"
__global__ void normCalc (float *d_A, float *d_B, int n) {
int col = blockIdx.x * blockDim.x + threadIdx.x;
__shared__ int row, mu, sigma;
if (col < n){
mu = (float)0.0;
for (row=0; row < n; row++)
mu += d_A[col*n+row];
mu /= (float) n;

__syncthreads();

sigma = (float)0.0;
for (row=0; row < n; row++)
sigma += powf(d_A[col*n+row] - mu, (float)2.0);
sigma /= (float) n;

__syncthreads();

sigma = sqrt((float)sigma);


for (row=0; row < n; row++) {
if (sigma == (float)0.0)
d_B[row*n+col] = (float)0.0;
else
d_B[row*n+col] = (d_A[col*n+row] - mu) / sigma;
}
}
}