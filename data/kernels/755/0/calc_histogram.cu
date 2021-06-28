#include "includes.h"
__global__ void calc_histogram(char* dbuff, unsigned int* dcount, unsigned int size, float stride) {

unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int start_pos = stride * index;
unsigned int stop_pos = start_pos + stride;
unsigned int lcount[10] = { 0 };

if (size < stop_pos) {
stop_pos = size;
}

for (unsigned int i = start_pos; i < stop_pos; i++) {
// Increment counter per occurances
if (dbuff[i] == '0') {
lcount[0] += 1;
} else if (dbuff[i] == '1') {
lcount[1] += 1;
} else if (dbuff[i] == '2') {
lcount[2] += 1;
} else if (dbuff[i] == '3') {
lcount[3] += 1;
} else if (dbuff[i] == '4') {
lcount[4] += 1;
} else if (dbuff[i] == '5') {
lcount[5] += 1;
} else if (dbuff[i] == '6') {
lcount[6] += 1;
} else if (dbuff[i] == '7') {
lcount[7] += 1;
} else if (dbuff[i] == '8') {
lcount[8] += 1;
} else if (dbuff[i] == '9') {
lcount[9] += 1;
}
}

__syncthreads();

dcount[0] += lcount[0];
dcount[1] += lcount[1];
dcount[2] += lcount[2];
dcount[3] += lcount[3];
dcount[4] += lcount[4];
dcount[5] += lcount[5];
dcount[6] += lcount[6];
dcount[7] += lcount[7];
dcount[8] += lcount[8];
dcount[9] += lcount[9];
}