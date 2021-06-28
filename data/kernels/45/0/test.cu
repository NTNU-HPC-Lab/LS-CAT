#include "includes.h"
#ifndef _KERNEL_H
#define _KERNEL_H
typedef struct Node {
int starting;
int no_of_edges;
}Node;



#endif
__global__ void test(Node* d_graph_nodes, int no_of_nodes) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < no_of_nodes) {
d_graph_nodes[tid].starting+=1;
}
}