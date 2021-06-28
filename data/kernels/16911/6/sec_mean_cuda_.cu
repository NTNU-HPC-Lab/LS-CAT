#include "includes.h"
__global__ void sec_mean_cuda_(int nProposal, int C, float *inp, int *offsets, float *out){
for(int p_id = blockIdx.x; p_id < nProposal; p_id += gridDim.x){
int start = offsets[p_id];
int end = offsets[p_id + 1];

float count = (float)(end - start);

for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
float mean = 0;
for(int i = start; i < end; i++){
mean += (inp[i * C + plane] / count);
}
out[p_id * C + plane] = mean;
}
}
}