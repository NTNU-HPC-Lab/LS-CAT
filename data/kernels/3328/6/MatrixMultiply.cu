#include "includes.h"
__global__ void MatrixMultiply(const float* A_elements, const float* B_elements,  float* C_elements, const int X, const int Y, const int Z)
{
int baseMatrixRow = blockIdx.y * blockDim.y + threadIdx.y;
int baseMatrixCol = blockIdx.x * blockDim.x + threadIdx.x;

int strideX = blockDim.x * gridDim.x;
int strideY = blockDim.y * gridDim.y;

__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

for (int iterY = 0; iterY < (Y + strideY - 1) / strideY; iterY++)
{
for (int iterX = 0; iterX < (X + strideX - 1)/ strideX; iterX++)
{
int matrixRow = baseMatrixRow + strideY * (iterY);
int matrixCol = baseMatrixCol + strideX * (iterX);

int blockRow = threadIdx.y;
int blockCol = threadIdx.x;

float Cvalue = 0;

for (int i = 0; i < ((X + TILE_SIZE - 1) / TILE_SIZE); ++i)
{

if((blockCol + i*TILE_SIZE) < X && matrixRow < Y)
As[blockRow][blockCol] = A_elements[matrixRow * X + blockCol + i*TILE_SIZE];
else
As[blockRow][blockCol] = 0;

if((blockRow + i*TILE_SIZE) < X && matrixCol < Z)
Bs[blockRow][blockCol] = B_elements[(blockRow + i*TILE_SIZE) * Z + matrixCol];
else
Bs[blockRow][blockCol] = 0;

//Synchronize threads
__syncthreads();

for (int j = 0; j < TILE_SIZE; ++j)
{
Cvalue += As[blockRow][j] * Bs[j][blockCol];
}

__syncthreads();
}
if (matrixRow < Y && matrixCol < Z) //Saving Final result into Matrix C
{
C_elements[matrixRow * Z + matrixCol] = Cvalue;
}
}
}
}