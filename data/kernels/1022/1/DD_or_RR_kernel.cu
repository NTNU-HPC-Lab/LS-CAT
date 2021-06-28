#include "includes.h"


// 2-point angular correlation

const int BLOCKSIZE = 256;
const int ROWSPERTHREAD = 256;

// Columns are D and rows are R

// All computation in single-precision

__global__ void DD_or_RR_kernel(int nCols, int nRows, float *arr, unsigned long long int *gHist) {

// The thread id on the x-axis and y-axis
//int x = blockIdx.y * ROWSPERTHREAD + blockIdx.x * blockDim.x + threadIdx.x;
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * ROWSPERTHREAD;

// If the column is inside the domain and the last row of the thread should be computed
if (x < nCols && y + ROWSPERTHREAD > x) {

// Shared histogram for the thread block
__shared__ unsigned int sHist[720];

// Thread number zero will initialize the shared memory
if (threadIdx.x == 0) {
for (int i = 0; i < 720; i++) {
sHist[i] = 0;
}
}

__syncthreads();

// Right ascension and declination in degrees for the current column
float asc1 = arr[x * 2];
float dec1 = arr[x * 2 + 1];

// Offset is at which row to start computing
int offset = max(x-y+1, 0);

// The amount of rows to be calculated is ROWSPERTHREAD or rows left in the domain, whichever is smaller
int nElements = min(nRows-y, ROWSPERTHREAD);

for (int j = offset; j < nElements; j++) {
// Right ascension and declination in degrees for the current row
float asc2 = arr[(y + j) * 2];
float dec2 = arr[(y + j) * 2 + 1];

// Compute the intermediate value
float tmp = sinf(dec1) * sinf(dec2) + cosf(dec1) * cosf(dec2) * cosf(asc1-asc2);

// Clamp it to -1, 1
tmp = fminf(tmp, 1.0f);
tmp = fmaxf(tmp, -1.0f);

// Compute the angle in radians
float radianResult = acosf(tmp);

// Convert to degrees
float degreeResult = radianResult * 180.0f/3.14159f;

// Compute the bin index
int resultIndex = floor(degreeResult * 4.0f);

// Increment the bin in the shared histogram
atomicAdd(&sHist[resultIndex], 2);
}

__syncthreads();

// Thread number zero will write the shared histogram to global device memory
if (threadIdx.x == 0) {
for (int i = 0; i < 720; i++) {
// Update the global histogram with the shared histogram
atomicAdd(&gHist[i], sHist[i]);
}
}
}
}