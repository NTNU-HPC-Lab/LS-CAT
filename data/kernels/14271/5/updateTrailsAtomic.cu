#include "includes.h"
/**
*  Project TACO: Parallel ACO algorithm for TSP
*  15-418 Parallel Algorithms - Final Project
*  Ivan Wang, Carl Lin
*/




#define MAX_THREADS 128

__device__ static inline int toIndex(int i, int j) {
return i * MAX_CITIES + j;
}
__global__ void updateTrailsAtomic(float *phero, int *paths, float *tourLengths)
{
int antId = blockIdx.x;
int from, to;

for (int i = 0; i < MAX_CITIES; i++) {
from = paths[toIndex(antId, i)];
if (i < MAX_CITIES - 1) {
to = paths[toIndex(antId, i+1)];
} else {
to = paths[toIndex(antId, 0)];
}

if (from < to) {
int tmp = from;
from = to;
to = tmp;
}
atomicAdd(&phero[toIndex(from, to)], QVAL / tourLengths[antId]);
}
}