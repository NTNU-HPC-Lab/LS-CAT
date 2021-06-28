#include "includes.h"
__global__ void find_all_sums_kernel(bool *mask, double *node_weight, int *neighbor, int *neighbor_start, double *sum_weight_result, int width, int height){
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int nid = y * width + x; // thread_index is node id
if (x < width && y < height && mask[nid]){
double sum = 0.0;
int end = min(neighbor_start[nid+1], neighbor_start[nid]+HUB_THREASHOLD+1); //+1 because HUB_THREASHOLD is out degree
for (int eid = neighbor_start[nid]; eid < end; eid++) { // this eid is just index of the neighbor in the neighbor array
sum += node_weight[neighbor[eid]];
}
sum_weight_result[nid] = sum;
}
}