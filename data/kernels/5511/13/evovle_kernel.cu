#include "includes.h"
__global__ void evovle_kernel(int N, char *oldGen, char *newGen, int *allzeros, int *change)
{
// Achieve indexng on 2D blocks
int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
// Thread calculates its global id
int id = ix * (N+2) + iy;

int neighbors;

if (ix <= N && iy <= N) {
neighbors = oldGen[id+(N+2)] + oldGen[id-(N+2)]     //lower upper
+ oldGen[id+1] + oldGen[id-1]           //right left
+ oldGen[id+(N+3)] + oldGen[id-(N+3)]   //diagonals
+ oldGen[id-(N+1)] + oldGen[id+(N+1)];

char cell  = oldGen[id];
newGen[id] = neighbors == 3 || (neighbors == 2 && cell); // Fill in the cells

// Terminating Checkings
if (newGen[id] != 0) (*allzeros)++;             // Check if all cells are dead
if (newGen[id] != oldGen[id]) (*change)++;      // Check if life stayed the same

}
}