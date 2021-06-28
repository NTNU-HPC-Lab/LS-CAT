#include "includes.h"


__global__ void Pi_GPU(float *x, float *y, int *totalCounts, int N) {
int idx = blockIdx.x * blockDim.x + threadIdx.x; // номер элемента
int threadCount = gridDim.x * blockDim.x; //cмещение

int countPoints = 0;
for (int i = idx; i < N; i += threadCount) {
if (x[i] * x[i] + y[i] * y[i] < 1) {
countPoints++;
}
}
atomicAdd(totalCounts, countPoints); // каждый поток суммирует в переменную
}