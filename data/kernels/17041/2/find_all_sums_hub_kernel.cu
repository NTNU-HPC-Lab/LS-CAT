#include "includes.h"




#define BLOCK_SIZE 16
#define HUB_BLOCK_SIZE 32

#define TRANSITION_PROB 0.02 * 0.1

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