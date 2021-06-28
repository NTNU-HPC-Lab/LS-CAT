#include "includes.h"
__global__ void find_all_sums_hub_kernel(int* hub, int nhub, float *node_weight, int *neighbor, int *neighbor_start, float *neighbor_accum_weight_result, float *sum_weight_result){
int x = blockIdx.x * blockDim.x + threadIdx.x;
if (x < nhub) {
int nid = hub[x];
float sum = 0.0;
for (int eid = neighbor_start[nid]; eid < neighbor_start[nid+1]; eid++) { // this eid is just index of the neighbor in the neighbor array
sum += node_weight[neighbor[eid]];
neighbor_accum_weight_result[eid] = sum;
}
sum_weight_result[nid] = sum;
}
}