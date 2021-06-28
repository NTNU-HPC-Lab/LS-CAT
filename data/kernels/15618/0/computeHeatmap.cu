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

__global__ void computeHeatmap(float* desiredAgentsX, float* desiredAgentsY, int n, int* heatmap, size_t heatmap_pitch, int* scaled_heatmap, size_t scaled_heatmap_pitch) {
// Block row and column
int blockRow = blockIdx.y;
int blockCol = blockIdx.x;

// Thread row and column block
int row = threadIdx.y;
int col = threadIdx.x;

// x, y coordinate
int x = blockCol * blockDim.x + col;
int y = blockRow * blockDim.y + row;

// fade heatmap
int* heatPoint = (int*)((char*)heatmap + y * heatmap_pitch) + x;
*heatPoint = (int)round((*heatPoint) * 0.80);

// pull desiredAgentxX and Y array from global to shared memory, only 1 thread will do it
extern __shared__ float desiredPosition[];

if (row == 0 && col == 0) {
for (int i = 0; i < n; i++) {
desiredPosition[i] = desiredAgentsX[i];
desiredPosition[i + n] = desiredAgentsY[i];
}
}

__syncthreads();

// Count how many agents want to go to each location
for (int i = 0; i < n; i++) {
int desiredX = (int)desiredPosition[i];
int desiredY = (int)desiredPosition[i + n];

if (x == desiredX && y == desiredY) {
// intensify heat for better color results
if ((*heatPoint) + 40 <= 255) {
*heatPoint += 40;
}
}
}
}