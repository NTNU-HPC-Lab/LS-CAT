#include "includes.h"
#pragma comment(lib,"cublas.lib")




using namespace std;

//==============================Function Prototypes================================
double getRand();

__global__ void initWeights(float *dst, unsigned int seed){
//params are: seed,sequence num,offset,handle
int n = blockIdx.x*blockDim.x + threadIdx.x;
dst[n] = dst[n]/(float)(seed);
while(dst[n] > 5) {
dst[n]=dst[n]/2;
}
if (n%(seed % 3) == 0) {
dst[n] = dst[n] * -1;
}
}