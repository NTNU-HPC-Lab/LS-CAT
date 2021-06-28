#include "includes.h"
__global__ void evovle_kernel_shared(int N, char *oldGen, char *newGen, int *allzeros, int *change)
{
// Global
int ix = (blockDim.x - 2) * blockIdx.x + threadIdx.x;       //Different indexing as we declared more blocks (see SideGrid)
int iy = (blockDim.y - 2) * blockIdx.y + threadIdx.y;
int id = ix * (N+2) + iy;

int i = threadIdx.x;
int j = threadIdx.y;
int neighbors;

// Declare the shared memory on a per block level
__shared__ char oldGen_shared[BLOCK_SIZE][BLOCK_SIZE];

// Copy cells into shared memory
if (ix <= N+1 && iy <= N+1)
oldGen_shared[i][j] = oldGen[id];           //Copy each cell and in the sides of shared array the blocks' neighbors

// Sync threads on block
__syncthreads();

if (ix <= N && iy <= N) {
if(i != 0 && i != (blockDim.y-1) && j != 0 && j != (blockDim.x-1)) {

// Get the number of neighbors for a given oldGen point
neighbors = oldGen_shared[i+1][j] + oldGen_shared[i-1][j]         //lower upper
+ oldGen_shared[i][j+1] + oldGen_shared[i][j-1]           //right left
+ oldGen_shared[i+1][j+1] + oldGen_shared[i-1][j-1]       //diagonals
+ oldGen_shared[i-1][j+1] + oldGen_shared[i+1][j-1];

char cell  = oldGen_shared[i][j];
newGen[id] = neighbors == 3 || (neighbors == 2 && cell); // Fill in  the cells

// Terminating Checkings
if (newGen[id] != 0) (*allzeros)++;        // Check if all cells are dead
if (newGen[id] != oldGen[id]) (*change)++; // Check if life stayed the same
}
}
}