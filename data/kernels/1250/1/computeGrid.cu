#include "includes.h"

#define SIZE 30000				//Length and width of inner grid in threads
#define DIM (SIZE + 2)			//Length and width of the entire grid in threads
#define GRID_SIZE 1500 			//Length and width of inner grid in blocks
#define BLOCK_SIZE 20 			//Length and width of block in threads
#define MEM_SIZE (sizeof(float) * DIM * DIM)
#define TIME_STEPS 1
#define PINNED 0

void fillGrid(float* grid);

__global__ void computeGrid(float* read, float* write) {
//Retrieve the thread's position in the grid
//The position is offset by 1 in the x and y directions to remove boundary checks
int x = blockDim.x * blockIdx.x + threadIdx.x + 1;
int y = blockDim.y * blockIdx.y + threadIdx.y + 1;

//Writes the sum of the neighbors to the cell
write[DIM * y + x] = read[DIM * (y - 1) + x] + read[DIM * (y + 1) + x] + read[DIM * y + x - 1] + read[DIM * y + x + 1];
}