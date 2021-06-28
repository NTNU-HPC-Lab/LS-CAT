#include "includes.h"
// System includes

// Other c++ libraries

// CUDA runtime

// Helper functions and utilities to work with CUDA

//#define N 2000
#define PI 3.141592653
#define PREC 20
#define maxNeighbors 6
#define maxNeighbors 6
typedef double4 particle;
typedef double dbl;
typedef double3 dbl3;
typedef double2 dbl2;
//typedef float4 particle;
//typedef float2 dbl2;
//typedef float3 dbl3;
//typedef float dbl;

using namespace std;

enum string_code {
enDim,
enumParticles,
ephi,
epotentialPower,
eisFinished,
enone
};

__global__ void LEShift(particle *parts, dbl LEshear) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
parts[i].y = parts[i].y + parts[i].x*LEshear;
return;
}