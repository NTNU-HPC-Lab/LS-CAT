#include "includes.h"
__global__ void kernel(int* D, int* Q, int bits){

// Find index
int i = blockIdx.x * blockDim.x + threadIdx.x;

// Initialize variables that will be shifted left and right
int shifted_right = i;
int shifted_left = shifted_right;

// Perform bit reversal permutation
for(int a = 1; a < bits; a++)
{
shifted_right >>= 1;
shifted_left <<= 1;
shifted_left |= shifted_right & 1;
}
shifted_left &= N - 1;

// Assign the values to the bit reversed positions
Q[shifted_left] = D[i];
}