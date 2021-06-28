#include "includes.h"
__global__ void get_iou_cuda_(int nInstance, int nProposal, int *proposals_idx, int *proposals_offset, long *instance_labels, int *instance_pointnum, float *proposals_iou){
for(int proposal_id = blockIdx.x; proposal_id < nProposal; proposal_id += gridDim.x){
int start = proposals_offset[proposal_id];
int end = proposals_offset[proposal_id + 1];
int proposal_total = end - start;
for(int instance_id = threadIdx.x; instance_id < nInstance; instance_id += blockDim.x){
int instance_total = instance_pointnum[instance_id];
int intersection = 0;
for(int i = start; i < end; i++){
int idx = proposals_idx[i];
if((int)instance_labels[idx] == instance_id){
intersection += 1;
}
}
proposals_iou[proposal_id * nInstance + instance_id] = (float)intersection / ((float)(proposal_total + instance_total - intersection) + 1e-5);
}
}
}