#include "includes.h"
/**
*  Project TACO: Parallel ACO algorithm for TSP
*  15-418 Parallel Algorithms - Final Project
*  Ivan Wang, Carl Lin
*/




#define MAX_THREADS 128

__global__ void copyBestPath(int i, int *bestPathResult, int *pathResults) {
memcpy(bestPathResult, &pathResults[i * MAX_ANTS], MAX_CITIES * sizeof(int));
}