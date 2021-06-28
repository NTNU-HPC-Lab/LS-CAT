#include "includes.h"
__global__ void add (float *d_A, float *d_B, float *d_C, int widthA, int widthB, int widthC)
{
int startA = blockIdx.x*64 + threadIdx.x*2 + (blockIdx.y*8 + threadIdx.y)*widthA;
int startB = blockIdx.x*64 + threadIdx.x*2 + (blockIdx.y*8 + threadIdx.y)*widthB;
int startC = blockIdx.x*64 + threadIdx.x*2 + (blockIdx.y*8 + threadIdx.y)*widthC;
float2 tempA = *(float2 *)(d_A+startA);
float2 tempB = *(float2 *)(d_B+startB);
tempA.x += tempB.x;
tempA.y += tempB.y;
*(float2 *)(d_C+startC) = tempA;
}