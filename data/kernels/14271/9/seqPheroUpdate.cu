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
__global__ void seqPheroUpdate(float *phero, float *pheroReal, int *paths, float *tourLengths) {
memcpy(phero, pheroReal, sizeof(float) * MAX_CITIES * MAX_CITIES);

int from, to;
// evaporate
for (from = 0; from < MAX_CITIES; from++) {
for (to = 0; to < from; to++) {
phero[toIndex(from, to)] *= 1.0 - RHO;

if (phero[toIndex(from, to)] < 0.0) {
phero[toIndex(from, to)] = INIT_PHER;
}
phero[toIndex(to, from)] = phero[toIndex(from, to)];
}
}

//Add new pheromone to the trails
for (int ant = 0; ant < MAX_ANTS; ant++) {
for (int i = 0; i < MAX_CITIES; i++) {
from = paths[toIndex(ant, i)];
if (i < MAX_CITIES - 1) {
to = paths[toIndex(ant, i+1)];
} else {
to = paths[toIndex(ant, 0)];
}

phero[toIndex(from, to)] += (QVAL / tourLengths[ant]);
phero[toIndex(to, from)] = phero[toIndex(from, to)];
}
}

}