#include "includes.h"
__global__ void sec_max_cuda_(int nProposal, int C, float *inp, int *offsets, float *out){
for(int p_id = blockIdx.x; p_id < nProposal; p_id += gridDim.x){
int start = offsets[p_id];
int end = offsets[p_id + 1];

for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
float max_val = -1e50;
for(int i = start; i < end; i++){
if(inp[i * C + plane] > max_val){
max_val = inp[i * C + plane];
}
}
out[p_id * C + plane] = max_val;
}
}
}