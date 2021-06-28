#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void roadCrossingsKernel(int rows, int segs, int* adjacency, int* cross) {

int idx = blockIdx.x*blockDim.x + threadIdx.x;

if (idx < rows) {
cross[idx] = 0;

for (int ii = 0; ii < segs; ii++) {
cross[idx] += adjacency[idx*segs + ii];
}
}
}