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
__global__ void checkPhero(float *pheroSeq, float *phero) {
for (int i = 0; i < MAX_CITIES; i++) {
for (int j = 0; j < MAX_CITIES; j++) {
if (i == j) continue;
int idx = toIndex(i, j);
if (fabsf(pheroSeq[idx] - phero[idx]) > 0.001) {
printf("PHERO IS BROKEN at (%d, %d); expected: %1.15f, actual: %1.15f\n", i, j, pheroSeq[idx], phero[idx]);
}
}
}
}