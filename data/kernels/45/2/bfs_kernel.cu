#include "includes.h"
#ifndef _KERNEL_H
#define _KERNEL_H
typedef struct Node {
int starting;
int no_of_edges;
}Node;



#endif
__global__ void bfs_kernel(Node* d_graph_nodes, int* d_edge_list, bool* d_graph_level, bool* d_graph_visited, int* d_cost, bool* loop, int no_of_nodes) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
//d_graph_level[tid] is true means the vertex in the current level
//is being visited
if (tid < no_of_nodes && d_graph_level[tid]) {
d_graph_level[tid] = false;
d_graph_visited[tid] = true;
for (int i = d_graph_nodes[tid].starting; i <
(d_graph_nodes[tid].no_of_edges +
d_graph_nodes[tid].starting); i++) {
int id = d_edge_list[i];
if (!d_graph_visited[id]) {
//calculate in which level the vertex is visited
d_cost[id] = d_cost[tid] + 1;
d_graph_level[id] = true;
//to make the loop continues
*loop = true;
}
}
}
}