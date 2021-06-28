#include "includes.h"


__global__ void BFS_UNIFIED(int source, int* edges, int* dest, int* label, int* visited, int *c_frontier_tail, int *c_frontier, int *p_frontier_tail, int *p_frontier)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < *p_frontier_tail)
{
int c_vertex = p_frontier[i];
for (int i = edges[c_vertex]; i < edges[c_vertex+1]; ++i)
{
int was_visited = atomicExch(visited + dest[i], 1);
if (!was_visited)
{
int old_tail = atomicAdd(c_frontier_tail, 1);
c_frontier[old_tail] = dest[i];
label[dest[i]] = label[c_vertex] + 1;
}
}

}
}