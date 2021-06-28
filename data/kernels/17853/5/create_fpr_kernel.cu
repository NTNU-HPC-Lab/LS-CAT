#include "includes.h"
__global__ void create_fpr_kernel(float* tpr, const int* unique_index, float* fpr, int num_selected, int num_total) {
float pos_cnt = tpr[num_selected - 1];
float neg_cnt = num_total - pos_cnt;
int gid_base = blockIdx.x * blockDim.x + threadIdx.x;
for (int gid = gid_base; gid < num_selected; gid += blockDim.x * gridDim.x) {
float tp = tpr[gid];
fpr[gid] = (1.0f + unique_index[gid] - tp) / neg_cnt;
tpr[gid] = tp / pos_cnt;
}
}