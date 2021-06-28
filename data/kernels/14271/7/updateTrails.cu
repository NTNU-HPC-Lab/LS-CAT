#include "includes.h"
/**
*  Project TACO: Parallel ACO algorithm for TSP
*  15-418 Parallel Algorithms - Final Project
*  Ivan Wang, Carl Lin
*/




#define MAX_THREADS 128

__device__ static inline int calculateTo(int i) {
//find least triangle number less than i
int row = (int)(-1 + (sqrt((float)(1 + 8 * i)))) >> 1;
int tnum = (row * (row + 1)) >> 1;
int remain = i - tnum;
return row - remain;
}
__device__ static inline int calculateFrom(int i) {
//find least triangle number less than i
int row = (int)(-1 + (sqrt((float)(1 + 8 * i)))) >> 1;
int tnum = (row * (row + 1)) >> 1;
int remain = i - tnum;
return MAX_CITIES - 1 - remain;
}
__device__ static inline int toIndex(int i, int j) {
return i * MAX_CITIES + j;
}
__global__ void updateTrails(float *phero, int *paths, float *tourLengths)
{
//__shared__ float localPaths[MAX_CITIES];

int numPhero = (NUM_EDGES + (blockDim.x * (MAX_ANTS * 2) - 1)) /
(blockDim.x * (MAX_ANTS * 2));
int blockStartPhero = numPhero * blockDim.x * blockIdx.x;
int from, to;

int cur_phero;
for (int i = 0; i < MAX_ANTS; i++) {
// For each ant, cache paths in shared memory
/*int tile;
if (startCityIndex + citiesPerThread >= MAX_CITIES) {
tile = MAX_CITIES - startCityIndex;
} else {
tile = citiesPerThread;
}
memcpy(&localPaths[startCityIndex], &paths[i * MAX_CITIES + startCityIndex], tile * sizeof(float));
*/
// TODO: figure out tiling
/*if (threadIdx.x == 0) {
memcpy(&localPaths, &paths[i * MAX_CITIES], MAX_CITIES * sizeof(float));
}

__syncthreads();
*/

for (int j = 0; j < numPhero; j++) {
cur_phero = blockStartPhero + j + numPhero * threadIdx.x;

if (cur_phero >= NUM_EDGES) {
break;
}

from = calculateFrom(cur_phero); //triangle number thing
to = calculateTo(cur_phero);

bool touched = false;
int checkTo;
int checkFrom;
for (int k = 0; k < MAX_CITIES; k++) {
checkFrom = paths[toIndex(i, k)];
if (k < MAX_CITIES - 1) {
checkTo = paths[toIndex(i, k + 1)];
} else {
checkTo = paths[toIndex(i, 0)];
}

if ((checkFrom == from && checkTo == to) ||
(checkFrom == to && checkTo == from))
{
touched = true;
break;
}
}

if (touched) {
int idx = toIndex(from, to);
phero[idx] += (QVAL / tourLengths[i]);
phero[toIndex(to, from)] = phero[idx];
}
}
//__syncthreads();
}
}