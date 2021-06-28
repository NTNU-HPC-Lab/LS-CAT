#include "includes.h"

#define WEIGHTSUM 273
#define BLOCK_SIZE 16

int * heatmap;
size_t heatmap_pitch;

int * scaled_heatmap;
size_t scaled_heatmap_pitch;

int * blurred_heatmap;
size_t blurred_heatmap_pitch;

float* d_desiredPositionX;
float* d_desiredPositionY;

__global__ void computeScaledHeatmap(int* heatmap, size_t heatmap_pitch, int* scaled_heatmap, size_t scaled_heatmap_pitch) {
// Block row and column
int blockRow = blockIdx.y;
int blockCol = blockIdx.x;

// Thread row and column block
int row = threadIdx.y;
int col = threadIdx.x;

// x, y coordinate
int x = blockCol * blockDim.x + col;
int y = blockRow * blockDim.y + row;

// Scale the data for visual representation
int value = *((int*)((char*)heatmap + y * heatmap_pitch) + x);
for (int r = 0; r < CELLSIZE; r++) {
int* row = (int*)((char*)scaled_heatmap + (r + y * CELLSIZE) * scaled_heatmap_pitch);
for (int c = 0; c < CELLSIZE; c++) {
row[x * CELLSIZE + c] = value;
}
}
}