#include "includes.h"
__global__ void sortAtomsGenCellListsAlt(unsigned int natoms, const float4 *xyzr_d, const float4 *color_d, const unsigned int *atomIndex_d, unsigned int *sorted_atomIndex_d, const unsigned int *atomHash_d, float4 *sorted_xyzr_d, float4 *sorted_color_d, uint2 *cellStartEnd_d) {
extern __shared__ unsigned int hash_s[]; // blockSize + 1 elements
unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
unsigned int hash;

// do nothing if current index exceeds the number of atoms
if (index < natoms) {
hash = atomHash_d[index];
hash_s[threadIdx.x+1] = hash; // use smem to avoid redundant loads
if (index > 0 && threadIdx.x == 0) {
// first thread in block must load neighbor particle hash
hash_s[0] = atomHash_d[index-1];
}
}

__syncthreads();

if (index < natoms) {
// Since atoms are sorted, if this atom has a different cell
// than its predecessor, it is the first atom in its cell, and
// it's index marks the end of the previous cell.
if (index == 0 || hash != hash_s[threadIdx.x]) {
cellStartEnd_d[hash].x = index; // set start
if (index > 0)
cellStartEnd_d[hash_s[threadIdx.x]].y = index; // set end
}

if (index == natoms - 1) {
cellStartEnd_d[hash].y = index + 1; // set end
}

// Reorder atoms according to sorted indices
unsigned int sortedIndex = atomIndex_d[index];
sorted_atomIndex_d[sortedIndex] = index;
float4 pos = xyzr_d[sortedIndex];
sorted_xyzr_d[index] = pos;

// Reorder colors according to sorted indices, if provided
if (color_d != NULL) {
float4 col = color_d[sortedIndex];
sorted_color_d[index] = col;
}
}
}