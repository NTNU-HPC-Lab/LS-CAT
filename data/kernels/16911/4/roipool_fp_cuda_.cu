#include "includes.h"
__global__ void roipool_fp_cuda_(int nProposal, int C, float *feats, int *proposals_offset, float *output_feats, int *output_maxidx){
for(int pp_id = blockIdx.x; pp_id < nProposal; pp_id += gridDim.x){
int start = proposals_offset[pp_id];
int end = proposals_offset[pp_id + 1];

for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
int argmax_idx = -1;
float max_val = -1e50;

for(int i = start; i < end; i++){
if(feats[i * C + plane] > max_val){
argmax_idx = i;
max_val = feats[i * C + plane];
}
}
output_maxidx[pp_id * C + plane] = argmax_idx;
output_feats[pp_id * C + plane] = max_val;
}
}
}