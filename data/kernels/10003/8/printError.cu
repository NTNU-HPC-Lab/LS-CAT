#include "includes.h"
#pragma comment(lib,"cublas.lib")




using namespace std;

//==============================Function Prototypes================================
double getRand();

__global__ void printError(float *output,float *target) {
int n = blockIdx.x*blockDim.x + threadIdx.x;
float error = target[n] - output[n];
printf("%f \n", error );
}