#include "includes.h"
__global__ void respond_kernel(int64_t *out, const int64_t *proposal, const int64_t *rowptr, const int64_t *col, int64_t numel) {
const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
if (thread_idx < numel) {
if (out[thread_idx] != -2)
return; // Only vist red nodes.

bool has_unmatched_neighbor = false;

for (int64_t i = rowptr[thread_idx]; i < rowptr[thread_idx + 1]; i++) {
auto v = col[i];

if (out[v] < 0)
has_unmatched_neighbor = true; // Unmatched neighbor found.

if (out[v] == -1 && proposal[v] == thread_idx) {
// Match first blue neighbhor v which proposed to u.
out[thread_idx] = min(thread_idx, v);
out[v] = min(thread_idx, v);
break;
}
}

if (!has_unmatched_neighbor)
out[thread_idx] = thread_idx;
}
}