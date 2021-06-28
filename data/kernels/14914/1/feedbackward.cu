#include "includes.h"


using namespace std;
__device__ __managed__ float weightedinputs[25]; // used as list of input neuron
__device__ __managed__ float weights[25]; // used as list of neuron conection weigths
__device__ __managed__ int inputs[25]; // used as list of neuron conection weigths
__device__ __managed__ float output = 0; // used to return output
__device__ __managed__ int expctd = 0; // used to return output



__global__ void feedbackward(){			// trains the weights
float lr = 0.3;
float error = (expctd - output);
weights[threadIdx.x] = weights[threadIdx.x] + error * inputs[threadIdx.x] * lr;
}