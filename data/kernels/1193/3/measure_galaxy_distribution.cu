#include "includes.h"

#define BIN_WIDTH 0.25
#define BLOCK_DIM 256
#define COVERAGE 180
#define LINE_LENGTH 30

#define BINS_TOTAL (COVERAGE * (int)(1 / BIN_WIDTH))

typedef struct Galaxy
{
float declination;
float declination_cos;
float declination_sin;
float right_ascension;
} Galaxy;


__global__ void measure_galaxy_distribution(int *DD_histogram, int *DR_histogram, int *RR_histogram, float *distribution, int n)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int i = index; i < n; i += stride)
{
if (RR_histogram[i] == 0)
continue;

distribution[i] = (DD_histogram[i] - 2.0f * DR_histogram[i] + RR_histogram[i]) / RR_histogram[i];
}
}