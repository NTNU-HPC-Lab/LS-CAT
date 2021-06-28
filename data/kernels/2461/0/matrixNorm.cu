#include "includes.h"
/* Matrix normalization.
* Compile with "nvcc matrixNormCuda.c -lm"
*/


/* Program Parameters */
#define N 8000  /* Matrix size */
int blocks_per_grid = 32;
int threads_per_block = 256;

/* Matrices */
float A[N*N], B[N*N];

/* CUDA arrays */
float *A_d, *B_d;


/* Initialize A and B*/
__global__ void matrixNorm(float* A_dd, float* B_dd, int N_d) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// __shared__ float mu, sigma;
float mu, sigma;
int row;

if (idx < N_d) {
mu = 0.0;
for (row=0; row < N_d; row++){
mu += A_dd[row*N_d + idx];
}
mu /= N_d;

sigma = 0.0;
for (row=0; row < N_d; row++){
sigma += powf(A_dd[row*N_d + idx] - mu, 2.0);
}
sigma /= N_d;
sigma = sqrt(sigma);

for (row=0; row < N_d; row++) {
if (sigma == 0.0){
B_dd[row*N_d + idx] = 0.0;
}
else{
B_dd[row*N_d + idx] = (A_dd[row*N_d + idx] - mu) / sigma;
}
}
}
}