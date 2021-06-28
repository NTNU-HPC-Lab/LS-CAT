#include "includes.h"
__global__ void device_BFS(const int* edges, const int* dests, int* labels, int* visited, int* c_frontier_tail, int* c_frontier, int* p_frontier_tail, int* p_frontier) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < *p_frontier_tail) {
int c_vertex = p_frontier[index];
for (int i = edges[c_vertex]; i < edges[c_vertex+1]; i++) {
int was_visited = atomicExch(visited + dests[i], 1);
if (!was_visited) {
int old_tail = atomicAdd(c_frontier_tail, 1);
c_frontier[old_tail] = dests[i];
labels[dests[i]] = labels[c_vertex] + 1;
}
}
}
}