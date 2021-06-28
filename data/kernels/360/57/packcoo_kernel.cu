#include "includes.h"
__global__ void packcoo_kernel(int num_entries, int* row_indices, int* column_indices, int* aggridx, int* partidx, int* partlabel)
{
int entryidx = blockIdx.x * blockDim.x + threadIdx.x;
if(entryidx < num_entries)
{
int row = row_indices[entryidx];
int col = column_indices[entryidx];
int l = partlabel[row];
int partstart = aggridx[partidx[l]];
unsigned int newindex = row - partstart;
newindex <<= 16;
newindex += col - partstart;
row_indices[entryidx] = newindex;
}
}