#include "includes.h"
__global__ void cudaising(int* G, double* w, int* newG) {

int index = threadIdx.x + blockIdx.x * blockDim.x;
double newSpin = 0.0;
for (int ii = -2; ii <= 2; ii++) {
for (int jj = -2; jj <= 2; jj++) {

newSpin += w[(jj + 2) + (ii + 2) * 5] * G[((jj + threadIdx.x + blockDim.x) % blockDim.x) + ((blockIdx.x + ii + blockDim.x) % blockDim.x) * blockDim.x];
}
}


if (newSpin > 0.000001) {
newG[index] = 1;
}
//if newSpin < 0 then the updated spin = -1
else if (newSpin < -0.000001) {
newG[index] = -1;
}
//if newSpin = 0 then the updated spin = old spin
else {
newG[index] = G[index];
}


}