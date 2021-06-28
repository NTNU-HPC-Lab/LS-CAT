#include "includes.h"
__device__ float_t d_randu(int * seed, int index) {

int M = INT_MAX;
int A = 1103515245;
int C = 12345;
int num = A * seed[index] + C;
seed[index] = num % M;

return fabsf(seed[index] / ((float_t) M));
}
__device__ void cdfCalc(float_t * CDF, float_t * weights, int Nparticles) {
int x;
CDF[0] = weights[0];
for (x = 1; x < Nparticles; x++) {
CDF[x] = weights[x] + CDF[x - 1];
}
}
__global__ void normalize_weights_kernel(float_t * weights, int Nparticles, float_t* partial_sums, float_t * CDF, float_t * u, int * seed) {
int block_id = blockIdx.x;
int i = blockDim.x * block_id + threadIdx.x;
__shared__ float_t u1, sumWeights;

if (0 == threadIdx.x)
sumWeights = partial_sums[0];

__syncthreads();

if (i < Nparticles) {
weights[i] = weights[i] / sumWeights;
}

__syncthreads();

if (i == 0) {
cdfCalc(CDF, weights, Nparticles);
u[0] = (1 / ((float_t) (Nparticles))) * d_randu(seed, i); // do this to allow all threads in all blocks to use the same u1
}

__syncthreads();

if (0 == threadIdx.x)
u1 = u[0];

__syncthreads();

if (i < Nparticles) {
u[i] = u1 + i / ((float_t) (Nparticles));
}
}