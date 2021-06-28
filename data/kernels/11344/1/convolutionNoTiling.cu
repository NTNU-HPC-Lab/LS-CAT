#include "includes.h"





#define _USE_MATH_DEFINES

static void CheckCudaErrorAux(const char *, unsigned, const char *,
cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
* Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
__global__ void convolutionNoTiling(float *I, float *P, int channels, int width, int height) {

int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int depth = threadIdx.z;

if (col < width && row < height && depth < channels) {

// Evaluate convolution
float pValue = 0;

int startRow = row - maskRowsRadius;
int startCol = col - maskColumnsRadius;

for (int i = 0; i < maskRows; i++) {
for (int j = 0; j < maskColumns; j++) {
int currentRow = startRow + i;
int currentCol = startCol + j;

float iValue;

// Check for ghost elements
if (currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width) {
iValue = I[(currentRow * width + currentCol) * channels + depth];
}
else {
iValue = 0.0f;
}

pValue += iValue * deviceMaskData[i * maskRows + j];
}
}

// Salva il risultato dal registro alla global
P[(row * width + col) * channels + depth] = pValue;
}
}