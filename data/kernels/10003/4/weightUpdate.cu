#include "includes.h"
#pragma comment(lib,"cublas.lib")




using namespace std;

//==============================Function Prototypes================================
double getRand();

__global__ void weightUpdate(float *d_W,float *d_D,float *d_N){
int2 pos;
pos.x = blockIdx.x*blockDim.x + threadIdx.x;//row j
pos.y = blockIdx.y*blockDim.y + threadIdx.y;//column k
int n = pos.x*blockDim.x*gridDim.y + pos.y;
float N = 0.1;
d_W[n] = d_W[n] + N*d_D[pos.y] * d_N[pos.x];
}