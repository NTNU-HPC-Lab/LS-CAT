#include "includes.h"
__global__ void roipool_bp_cuda_(int nProposal, int C, float *d_feats, int *proposals_offset, int *output_maxidx, float *d_output_feats){
for(int pp_id = blockIdx.x; pp_id < nProposal; pp_id += gridDim.x){
for(int plane = threadIdx.x; plane < C; plane += blockDim.x){
int argmax_idx = output_maxidx[pp_id * C + plane];
atomicAdd(&d_feats[argmax_idx * C + plane], d_output_feats[pp_id * C + plane]);
}
}
}