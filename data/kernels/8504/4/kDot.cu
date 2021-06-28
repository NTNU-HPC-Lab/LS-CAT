#include "includes.h"
__global__ void kDot(const int nThreads, const float *m1, const float *m2, float *output, const int m1_rows, const int m1_columns, const int m2_columns) {

for (int i = blockIdx.x * blockDim.x + threadIdx.x;
i < nThreads;
i += blockDim.x * gridDim.x)
{
int r = (int)i / m2_columns;
int c = i % m2_columns;
float t_output = 0.f;

for (int k = 0; k < m1_columns; ++k) {
t_output += m1[r * m1_columns + k] * m2[k * m2_columns + c];
}

output[i] = t_output;
}
}