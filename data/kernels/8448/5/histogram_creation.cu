#include "includes.h"
__device__ __host__ void print(float *result)
{
for(int k = 0; k < 3; k++)
{
for(int i = 0; i < N; i++)
{
for(int j = 0; j < N; j++)
printf("%f ",result[(i*N + j)*3 + k]);
printf("\n");
}
printf("\n");
}
}
__global__ void histogram_creation(int *A, int *hist, int no_of_threads) {

int global_x = blockDim.x * blockIdx.x + threadIdx.x;
__shared__ int local_hist[N+1];

for(int i = threadIdx.x; i<=N; i = i + (blockDim.x ) ){
local_hist[i] = 0;
}
__syncthreads();

for(int i = global_x; i <= M; i = i + (blockDim.x * no_of_threads)) {
atomicAdd(&local_hist[A[i]],1);
}
__syncthreads();

for(int i = threadIdx.x ; i <= N; i = i + (blockDim.x) ) {
atomicAdd(&hist[i],local_hist[i]);
printf("%d histogram_local %d \n",local_hist[i],i);
}
__syncthreads();

}