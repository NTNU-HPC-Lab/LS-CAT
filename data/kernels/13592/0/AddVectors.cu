#include "includes.h"
// Lab2_AddingTwoVectors.cu : Defines the entry point for the console application.
// Author: £ukasz Pawe³ Rabiec (259049)


#define SIZE 32

__global__ void AddVectors(int* a, int* b, int* c)
{
int tid = blockIdx.x;

if (tid < SIZE)
{
c[tid] = a[tid] + b[tid];
}

}