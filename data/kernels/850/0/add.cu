#include "includes.h"
/**
*	This is my first program in learning parallel programming using CUDA.
*	Equivalent to a hello World program :-)
*	This program basically performs two tasks:
*	1. It selects suitable CUDA enabled device(GPU) and prints the device properties
*	2. It demonstrate basic parallel addition of two arrays on the device(GPU) using add kernel.
*	Author: Shubham Singh
**/


#define N 10						/*N is size of arrays*/

using namespace std;

/************************************************************************************************************
*	Function:	Kernel to perform addition of two arrays in parallel on device(GPU)
*	Input:		Takes 3 pointer to int variables pointing to some memory locations on the device(GPU)
*	Output:		None
************************************************************************************************************/

__global__ void add(int *a, int *b, int *c)
{
int i = blockIdx.x;				/*blockIDx.x holds ID of block and acts as index*/
if (i < N)
c[i] = a[i] + b[i];
}