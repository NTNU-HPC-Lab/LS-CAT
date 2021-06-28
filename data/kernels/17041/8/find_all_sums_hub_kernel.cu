#include "includes.h"
__global__ void find_all_sums_hub_kernel(int* hub, int nhub, double *node_weight, int *neighbor, int *neighbor_start, double *sum_weight_result){
int x = blockIdx.x * blockDim.x + threadIdx.x;
if (x < nhub) {
int nid = hub[x];
double sum = 0.0;
for (int eid = neighbor_start[nid]; eid < neighbor_start[nid+1]; eid++) { // this eid is just index of the neighbor in the neighbor array
sum += node_weight[neighbor[eid]];
}
sum_weight_result[nid] = sum;
}
}