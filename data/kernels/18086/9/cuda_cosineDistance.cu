#include "includes.h"
__global__ void cuda_cosineDistance(double *x, double* y, int64_t len, double *dot_product, double *norm_x, double*norm_y)
{
int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;

int64_t cacheIdx = threadIdx.x;

__shared__ double dot_cache[threadsPerBlock];
__shared__ double norm_x_cache[threadsPerBlock];
__shared__ double norm_y_cache[threadsPerBlock];

double dot_tmp = 0;
double norm_x_tmp = 0;
double norm_y_tmp = 0;

while(idx < len)
{
dot_tmp += x[idx] * y[idx];
norm_x_tmp += x[idx] * x[idx];
norm_y_tmp += y[idx] * y[idx];
idx += blockDim.x * gridDim.x;
}
dot_cache[cacheIdx] = dot_tmp;
norm_x_cache[cacheIdx] = norm_x_tmp;
norm_y_cache[cacheIdx] = norm_y_tmp;
__syncthreads();

int64_t i = blockDim.x/2;
while(i!=0)
{
if(cacheIdx < i)
{
dot_cache[cacheIdx] += dot_cache[cacheIdx + i];
norm_x_cache[cacheIdx] += norm_x_cache[cacheIdx + i];
norm_y_cache[cacheIdx] += norm_y_cache[cacheIdx + i];
}
__syncthreads();
i/=2;
}

if(cacheIdx == 0)
{
dot_product[blockIdx.x] = dot_cache[0];
norm_x[blockIdx.x] = norm_x_cache[0];
norm_y[blockIdx.x] = norm_y_cache[0];
}
}