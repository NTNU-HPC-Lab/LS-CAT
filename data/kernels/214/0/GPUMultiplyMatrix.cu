#include "includes.h"

const int NUMTHREADS = 1024;
int startNodeNumber;
int endNodeNumber;

typedef struct lList {
int path[50];
struct lList *next;
} lList;

__global__ void GPUMultiplyMatrix(long *matrix1, long *matrix2, int paths, int count) {
int element = blockIdx.x * blockDim.x + threadIdx.x;
int i;
while (paths > 0) {
long sum = 0;
int col = element % count;
int row = element / count;
for (i = 0; i < count; i++) {
sum += matrix1[count * i + col] * matrix2[row * count + i];
}
//Wait till all GPU cores are finished
__syncthreads();
matrix2[element] = sum;

paths--;
}
}