#include "includes.h"
__global__ void instance_iou_cuda_kernel( int64_t total_gt_instances, const int64_t* __restrict__ nInstance, int nProposal, const int64_t* __restrict__ proposals_idx, const int64_t* __restrict__ proposals_offset, const int64_t* __restrict__ instance_labels, const int64_t* __restrict__ offset_num_gt_instances, const int64_t* __restrict__ batch, const int64_t* __restrict__ instance_pointnum, float* proposals_iou)
{
for (int proposal_id = blockIdx.x; proposal_id < nProposal; proposal_id += gridDim.x)
{
int start = proposals_offset[proposal_id];
int end = proposals_offset[proposal_id + 1];
int sampleIdx = batch[proposals_idx[start]];
int sampleNInstances = nInstance[sampleIdx];
int instanceOffset = offset_num_gt_instances[sampleIdx];
int proposal_total = end - start;
for (int instance_id = threadIdx.x; instance_id < sampleNInstances;
instance_id += blockDim.x)
{
int instance_total = instance_pointnum[instanceOffset + instance_id];
int intersection = 0;
for (int i = start; i < end; i++)
{
int idx = proposals_idx[i];
if ((int)instance_labels[idx] == instance_id + 1)
{ // 0 is reserved for "no instance"
intersection += 1;
}
}

proposals_iou[instanceOffset + instance_id + proposal_id * total_gt_instances] =
(float)intersection /
((float)(proposal_total + instance_total - intersection) + 1e-5);
}
}
}