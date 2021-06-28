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


__device__ float arcminutes_to_radians(float arcminute_value)
{
return (M_PI * arcminute_value) / (60 * 180);
}
__global__ void adjust_galaxy_set(Galaxy *galaxy_set, int n)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int i = index; i < n; i += stride)
{
float declination = arcminutes_to_radians(galaxy_set[i].declination);
galaxy_set[i].declination = declination;
galaxy_set[i].declination_cos = cosf(declination);
galaxy_set[i].declination_sin = sinf(declination);

galaxy_set[i].right_ascension = arcminutes_to_radians(galaxy_set[i].right_ascension);
}
}