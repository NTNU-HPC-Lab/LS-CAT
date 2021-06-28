#include "includes.h"
__global__ void cuda_graph_maxpool_bprop(float* gradInput, const float *gradOutput, const float* indices, const int nClusters, const int dim, const int nClustersPerThread) {

extern __shared__ float shared_mem[];
float* gradOutput_data = (float*)shared_mem;
float* indices_data = (float*)&gradOutput_data[nClusters];

const int tidx = threadIdx.x;
gradInput += blockIdx.x * dim;
gradOutput += blockIdx.x * nClusters;
indices += blockIdx.x * nClusters;
__syncthreads();
for (int i = 0; i < nClustersPerThread; ++i) {
int idx = tidx + i*blockDim.x;
if (idx < nClusters) {
gradOutput_data[idx] = gradOutput[idx];
indices_data[idx] = indices[idx];
}
}
__syncthreads();

//ouch...
if (tidx == 1) {
for (int i = 0; i < nClusters; ++i) {
gradInput[(int)indices_data[i]-1] += gradOutput[i];
}
}
//gradInput[(int)indices_data[tidx]-1] = gradOutput[tidx];
}