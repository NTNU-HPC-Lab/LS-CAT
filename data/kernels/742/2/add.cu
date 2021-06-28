#include "includes.h"
__global__ void add(int *a, int *r, int *g, int *b, float *gc)
{

int i = (blockIdx.x*blockDim.x) + threadIdx.x;

gc[5120 * 6 + i * 6    ] = b[i] * 0.00390625;
//gc[5120 * 6 + i * 6    ] = float(b[i]) / 256;
gc[5120 * 6 + i * 6 + 1] = g[i] * 0.00390625;
//gc[5120 * 6 + i * 6 + 1] = float(g[i]) / 256;
gc[5120 * 6 + i * 6 + 2] = r[i] * 0.00390625;
//gc[5120 * 6 + i * 6 + 2] = float(r[i]) / 256;

//	gc[5120 * 6 + i * 6 + 3] = float(i - ((i>>9)<<9) );  // i%512
//gc[5120 * 6 + i * 6 + 3] = float(i % 512);
//	gc[5120 * 6 + i * 6 + 4] = float( i >> 9);
//gc[5120 * 6 + i * 6 + 4] = float((i - (i % 512)) / 512);
//	gc[5120 * 6 + i * 6 + 5] = float(a[i]);
}