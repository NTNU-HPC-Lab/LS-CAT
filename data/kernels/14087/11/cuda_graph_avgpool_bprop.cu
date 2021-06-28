#include "includes.h"
__global__ void cuda_graph_avgpool_bprop(float* gradInput, const float *gradOutput, const float* clusters, const int nClusters, const int poolsize, const int dim, const int nClustersPerThread) {

extern __shared__ float shared_mem[];
float* gradOutput_data = (float*)shared_mem;

const int tidx = threadIdx.x;
gradInput += blockIdx.x * dim;
gradOutput += blockIdx.x * nClusters;
__syncthreads();
for (int i = 0; i < nClustersPerThread; ++i) {
int idx = tidx + i*blockDim.x;
if (idx < nClusters) {
gradOutput_data[idx] = gradOutput[idx];
}
}
__syncthreads();


if (tidx < poolsize) {
for (int i = 0; i < nClusters; ++i) {
gradInput[(int)(clusters[i*poolsize+tidx]-1)] += gradOutput[i]/poolsize;
}
}

/*
for (int j = 0; j < poolsize; ++j) {
gradInput[(int)(clusters[tidx*poolsize+j]-1)] += gradOutput[tidx]/poolsize;
__syncthreads();
}
*/
__syncthreads();

/*
//ouch...
if (tidx == 1) {
for (int i = 0; i < nClusters; ++i) {
//    int idx = tidx + i*blockDim.x;
for (int j = 0; j < poolsize; ++j) {
gradInput[(int)(clusters[i*poolsize+j]-1)] += gradOutput[i]/poolsize;
}
}
}
*/




/*
for (int i = 0; i < nClustersPerThread; ++i) {
int idx = tidx + i*blockDim.x;
if (idx < nClusters) {
for (int j = 0; j < poolsize; ++j) {
gradInput[(int)clusters[idx*poolsize+j]] += gradOutput_data[idx]/poolsize;
}
}
}
*/
}