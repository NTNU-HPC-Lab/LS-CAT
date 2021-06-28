#include "includes.h"
__global__ void bfsCheck( bool *d_graph_mask, bool *d_updating_graph_mask, bool *d_graph_visited, int no_of_nodes, bool *stop )
{
*stop = false;
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid < no_of_nodes){
if (d_updating_graph_mask[tid] == true){
d_graph_mask[tid] = true;
d_graph_visited[tid] = true;
*stop = true;
d_updating_graph_mask[tid] = false;
}
}
}