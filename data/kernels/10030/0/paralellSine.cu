#include "includes.h"
//
// Assignment 1: ParallelSine
// CSCI 415: Networking and Parallel Computation
// Spring 2017
// Name(s): Jaron Pollman
//
// Sine implementation derived from slides here: http://15418.courses.cs.cmu.edu/spring2016/lecture/basicarch


// standard imports

// problem size (vector length) N
static const int N = 12345678; //#of threads?

// Number of terms to use when approximating sine
static const int TERMS = 6; //# of blocks

// kernel function (CPU - Do not modify)
__global__ void paralellSine(float *input, float *output)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x; //Proper indexing of elements.
float value = input[idx];
float numer = input[idx] * input[idx] * input[idx];
int denom = 6;
int sign = -1;

for (int j=1; j<=TERMS; j++)
{
value += sign * numer/denom;
numer *= input[idx] * input[idx];
denom *= (2 * j + 2) * (2 * j + 3);
sign *= -1;
}
output[idx] = value;


}