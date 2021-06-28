#include "includes.h"

using namespace std;

#define MAX_ARRAY_SIZE 1024
#define RANDOM_MAX  1000
#define TILE_DIM 16
#define BLOCK_ROWS 8
#define EPSILON 0.000001
#define NUM_BLOCKS (MAX_ARRAY_SIZE/TILE_DIM)

float A[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];
float C[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];

void serial();
void init_F();
int check();

__global__ void matrixTranspose2(const float *F, float *C)
{
__shared__ float tile[TILE_DIM][TILE_DIM];

int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int width = gridDim.x * TILE_DIM;

for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
tile[threadIdx.y+j][threadIdx.x] = F[(y+j)*width + x];

__syncthreads();

x = blockIdx.y * TILE_DIM + threadIdx.x;
y = blockIdx.x * TILE_DIM + threadIdx.y;

for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
C[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}