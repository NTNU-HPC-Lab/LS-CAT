#include "includes.h"

#define COUNTERS 66
#define C_SIZE 64
#define C_STOP 65 // == C_SIZE+1
#define N 4224 // == COUNTERS*C_SIZE
#define N2 17842176 // == N*N

#define CUDA_ERROR_CHECK

#define cudaSafeCall(error) __cudaSafeCall(error, __FILE__, __LINE__)
#define cudaCheckErrors() __cudaCheckErrors(__FILE__, __LINE__)

__device__ inline int uniq(const int* M, int i, int* counters) {
for (int j = 1; j <= i - 1; j++) {
int a = (j - 1) * C_SIZE + counters[j-1];
int b = (i - 1) * C_SIZE + counters[i-1];
if (M[(a - 1) + N * (b - 1)] == 0)
return 1;
}
return 0;
}
__global__ void searcher(const int* M, int* res, size_t* itersNum) {
int partNumber = threadIdx.x + blockIdx.x * blockDim.x;
// initialize counters vector
int counters[COUNTERS];
for (int i = 0; i < COUNTERS; i++)
counters[i] = 1;

// go to selected part
counters[0] = 25;
counters[1] = 5;
counters[2] = 1;
counters[3] = 3;
counters[4] = 4;
counters[5] = 7;
counters[6] = 9;
counters[7] = 2;
counters[8] = 10;
counters[9] = 8;
counters[10] = (partNumber - 1) / 64 + 1;
counters[11] = (partNumber - 1) % 64 + 1;

size_t iter = 0;
size_t current = 1;
while (1) {
iter++;

// stop if search in the selected part is finished
if (counters[10] != (partNumber - 1) / 64 + 1 || counters[11] != (partNumber - 1) % 64 + 1) {
for (int i = 0; i < COUNTERS; i++)
res[partNumber * COUNTERS + i] = -1;
itersNum[partNumber] = iter;
break;
}

// first subspace is always good
if (current == 1)
current = 2;

// print intermediate state
// if (current == 13 && iter > 1000) {
//     fprintf(f, "Current state of part number %d:", partNumber);
//     for (int i = 0; i < COUNTERS; i++)
//         fprintf(f, " %d", counters[i]);
//     fprintf(f, "\nNumber of iterations: %f\n\n", iter);
//     fflush(f);
// }

for (int i = current; i <= COUNTERS; i++) {
if (uniq(M, i, counters) == 1) {
counters[i-1]++;
current = i;
while (counters[current-1] == C_STOP) {
counters[current - 1] = 1;
counters[current - 2] = counters[current - 2] + 1;
current--;
}
break;
}
}

if (current == COUNTERS && uniq(M, current, counters) == 0) {
for (int i = 0; i < COUNTERS; i++)
res[partNumber * COUNTERS + i] = counters[i];
itersNum[partNumber] = iter;
break;
}
}
}