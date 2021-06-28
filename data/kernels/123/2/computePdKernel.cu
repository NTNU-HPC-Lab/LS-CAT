#include "includes.h"
__device__ void sumByReduction( volatile double* sdata, double mySum, const unsigned int tid )
{
sdata[tid] = mySum;
__syncthreads();

// do reduction in shared mem
if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();
if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads();

if (tid < 32)
{
sdata[tid] = mySum = mySum + sdata[tid + 32];
sdata[tid] = mySum = mySum + sdata[tid + 16];
sdata[tid] = mySum = mySum + sdata[tid +  8];
sdata[tid] = mySum = mySum + sdata[tid +  4];
sdata[tid] = mySum = mySum + sdata[tid +  2];
sdata[tid] = mySum = mySum + sdata[tid +  1];
}
__syncthreads() ;
}
__global__ void computePdKernel(double* particle_pd, int particles_per_feature, int n_features, double* feature_pd)
{
__shared__ double shmem[256] ;
for ( int n = blockIdx.x ; n < n_features ;n+= gridDim.x ){
int offset = n*particles_per_feature ;
double val = 0 ;
for ( int i = offset+threadIdx.x ; i < offset + particles_per_feature ; i+= blockDim.x ){
val += particle_pd[i] ;
}
sumByReduction(shmem,val,threadIdx.x);

if ( threadIdx.x == 0)
feature_pd[n] = shmem[0]/particles_per_feature ;
__syncthreads() ;
}
}