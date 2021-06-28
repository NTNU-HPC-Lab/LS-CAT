#include "includes.h"

#define N 128


__global__ void calc_freq(int *freq, int file_size, char *buffer, int total_threads){
int temp[N];
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// Initialize temp with value 0
for (int i = 0; i < N; i++){
temp[i] = 0;
}

// Do the calculation
for(int i = idx; i < file_size; i += total_threads) {
temp[buffer[i]]++;
}

// Add the results from the threads to the blocks
for(int i = 0; i < N; i++){
atomicAdd(&freq[i], temp[i]);
}

}