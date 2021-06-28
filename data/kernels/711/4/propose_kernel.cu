#include "includes.h"
__global__ void propose_kernel(int64_t *out, int64_t *proposal, const int64_t *rowptr, const int64_t *col, int64_t numel) {

const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
if (thread_idx < numel) {
if (out[thread_idx] != -1)
return; // Only vist blue nodes.

bool has_unmatched_neighbor = false;

for (int64_t i = rowptr[thread_idx]; i < rowptr[thread_idx + 1]; i++) {
auto v = col[i];

if (out[v] < 0)
has_unmatched_neighbor = true; // Unmatched neighbor found.

if (out[v] == -2) {
proposal[thread_idx] = v; // Propose to first red neighbor.
break;
}
}

if (!has_unmatched_neighbor)
out[thread_idx] = thread_idx;
}
}