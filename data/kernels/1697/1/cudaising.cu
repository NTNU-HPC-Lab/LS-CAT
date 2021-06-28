#include "includes.h"
__global__ void cudaising(int* G, double* w, int* newG, int n, int workperthread) {

int startingId = threadIdx.x * workperthread;

//shared w and G in block
__shared__ double tempW[5 * 5];
__shared__ int tempG[(517 + 4) * 5];

//copy necessary elements from G into tempG
for (int i = -2; i <= 2; i++) {
for (int j = -2; j <= n + 2; j++) {
tempG[(j + 2) + (i + 2) * (n + 4)] = G[((j + n) % n) + ((blockIdx.x + i + n) % n) * n];
}
}


//copy using threads
/*if (threadIdx.x >=25&&threadIdx.x <30) {
for (int j = -2; j <= n + 2; j++) {
tempG[(j + 2) + (threadIdx.x-2-25 + 2) * (n + 4)] = G[((j + n) % n) + ((blockIdx.x + threadIdx.x-2-25 + n) % n) * n];

}
}
*/


//Copy w in tempW


if (threadIdx.x < 25) {
tempW[threadIdx.x] = w[threadIdx.x];
}
__syncthreads();




//for every element computed by this thread
for (int element = 0; element < workperthread; element++) {

double newSpin = 0.0;

//for every point in matrix w
for (int ii = 0; ii < 5; ii++) {
for (int jj = 0; jj < 5; jj++) {

//compute new Spin of element
newSpin += tempW[(jj)+(ii) * 5] * tempG[startingId + element + jj + ii * (n + 4)];

}
}
//global index of element whose spin was just calculated
int index = startingId + element + blockIdx.x * blockDim.x * workperthread;
//if newSpin > 0 then the updated spin = 1
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
__syncthreads();
}