#include "includes.h"
__global__ void next_move_kernel(bool *mask, int *rat_count, int *healthy_rat_count, int *exposed_rat_count, int *infectious_rat_count, double *node_weight, double *sum_weight_result,int *neighbor, int *neighbor_start,  int width, int height, double batch_fraction){
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int nid = y * width + x;
if (x < width && y < height){
for (int eid = neighbor_start[nid]; eid < neighbor_start[nid] + 5; eid++) { // 5 because self + up down left right
int remote_node = neighbor[eid];
double move_prob = batch_fraction * node_weight[remote_node] / sum_weight_result[nid]; // check 0
int move_rat = rat_count[nid] * move_prob;
int move_healthy = healthy_rat_count[nid] * move_prob;
int move_exposed = exposed_rat_count[nid] * move_prob;
int move_infectious = infectious_rat_count[nid] * move_prob;
atomicAdd(&rat_count[remote_node], move_rat);
atomicAdd(&healthy_rat_count[remote_node], move_healthy);
atomicAdd(&exposed_rat_count[remote_node], move_exposed);
atomicAdd(&infectious_rat_count[remote_node], move_infectious);
rat_count[nid] -= move_rat;
healthy_rat_count[nid] -= move_healthy;
exposed_rat_count[nid] -= move_exposed;
infectious_rat_count[nid] -= move_infectious;
}
}
}