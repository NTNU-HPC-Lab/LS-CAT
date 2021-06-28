#include "includes.h"
__global__ void Tri_k(float* a, float* b, float* c, float norm, int n)
{
// Identifies the thread working within a group
int tidx = threadIdx.x % n;
// Identifies the data concerned by the computations
int Qt = (threadIdx.x - tidx) / n;
// The global memory access index
int gb_index_x = Qt + blockIdx.x * (blockDim.x / n);

b[gb_index_x * n + tidx] = ((float)tidx + 1.0f) / (norm);
if (tidx > 0 && tidx < n - 1) {
a[gb_index_x * n + tidx] = ((float)tidx + 1.0f) / (norm * 3);
c[gb_index_x * n + tidx] = ((float)tidx + 1.0f) / (norm * 3);
}
else if (tidx == 0) {
a[gb_index_x * n + tidx] = 0.0f;
}
else {
c[gb_index_x * n + tidx] = 0.0f;
}
}