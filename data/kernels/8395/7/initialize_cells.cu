#include "includes.h"
__global__ void initialize_cells(CellT* dev_cells, CellT* dev_next_cells, int size_x, int size_y) {
for (int i = threadIdx.x + blockDim.x * blockIdx.x;
i < size_x*size_y; i += blockDim.x * gridDim.x) {
dev_cells[i] = 0;
dev_next_cells[i] = 0;
}
}