#include "includes.h"
#define CUDA_INPUT "input.txt"
#define CUDA_OUTPUT "cuda_output.txt"

int NUMPOINTS;
double ENDTIME;
double DT;
double ENDVALUES;

void InitialiseToZero(float* array);
__device__ void PrintPointsGPU(float* array, int size, double currentTime);
void PrintPointsCPU(float* array, double currentTime);
void ProcessOutput(float* array, int testCase, float time);
void CheckPoints(float* firstArray, float* secondArray);


__global__ void DiffuseHeat(float* currentPoints, float* nextPoints, const size_t size, double dx, double dt, const size_t endTime)
{
unsigned int threadIndex = (threadIdx.x + blockDim.x * blockIdx.x) + 1;
double currentTime = 0.0;
if (threadIndex > 0 && threadIndex < size-1)
{
while (currentTime < endTime)
{
nextPoints[threadIndex] = currentPoints[threadIndex] + 0.25*(currentPoints[threadIndex+1] - (2*currentPoints[threadIndex]) + currentPoints[threadIndex-1]);
__syncthreads();
currentPoints[threadIndex] = nextPoints[threadIndex];
currentTime += dt;
__syncthreads();
}
}
}