#include "includes.h"
extern "C" {
}



#define TB 256
#define EPS 0.1

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))


__global__ void histogram_kernel( float *I, float *minI, float *maxI, float *mask, int nbins, int c, int h, int w, float *hist )
{
int _id = blockIdx.x * blockDim.x + threadIdx.x;
int size = h * w;

if (_id < c * size) {
int id = _id % size, dc = _id / size;

if (mask[id] < EPS)
return ;

float val  = I[_id];

float _minI = minI[dc];
float _maxI = maxI[dc];


if (_minI == _maxI) {
_minI -= 1;
_maxI += 1;
}

if (_minI <= val && val <= _maxI) {
int idx = MIN((val - _minI) / (_maxI - _minI) * nbins, nbins-1);
int index = dc * nbins + idx;
atomicAdd(&hist[index], 1.0f);
}

}

return ;
}