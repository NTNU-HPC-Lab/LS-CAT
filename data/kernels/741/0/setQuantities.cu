#include "includes.h"



#define VERBOSE 0
#define INTEGER_SCALE_FACTOR 100

// Command line argument definitions
#define DEFAULT_NUM_REPEATS 1
#define DEFAULT_NUM_ITERATIONS 1
#define DEFAULT_NUM_ELEMENTS 128
#define DEFAULT_SEED 0
#define DEFAULT_DEVICE 0

#define MIN_ARGS 1
#define MAX_ARGS 6

#define ARG_EXECUTABLE 0
#define ARG_REPEATS 1
#define ARG_ITERATIONS 2
#define ARG_ELEMENTS 3
#define ARG_SEED 4
#define ARG_DEVICE 5

#define MAX 10

// Lazy CUDA Error handling
__global__ void setQuantities( unsigned int numInputs, unsigned int value, unsigned int * d_quantity ){
unsigned int tid = threadIdx.x + (blockDim.x * blockIdx.x);
if (tid < numInputs){
d_quantity[tid] = value;
}

}