#include "includes.h"
__global__ void cuda_dot(int N, double *a, double *b, double *c)
{
// __shared__ double localDot[threadsPerBlock];  /* Statically defined */
extern __shared__ double localDot[];
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int localIndex = threadIdx.x;

double localSum = 0;
while (ix < N)
{
localSum += a[ix] * b[ix];  /* Reduction is here */
ix += blockDim.x * gridDim.x;
}

/* Store sum computed by this thread */
localDot[localIndex] = localSum;

/* Wait for all threads to get to this point */
__syncthreads();

/* Every block should add up sum computed on
threads in the block */
int i = blockDim.x/2;
while (i != 0)
{
if (localIndex < i)
{
localDot[localIndex] += localDot[localIndex + i];
}
__syncthreads();
i /= 2;
}

/* Each block stores local dot product */
if (localIndex == 0)
c[blockIdx.x] = localDot[0];
}