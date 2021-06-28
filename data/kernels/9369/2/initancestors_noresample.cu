#include "includes.h"
__global__ void initancestors_noresample(int *ancestor, int np) {
int ii = threadIdx.x + blockIdx.x * BLOCKSIZE;
while (ii < np) {
ancestor[ii] = ii; //note that the next time step is the same as K time steps back. it's ok to overwrite this since we've already copied out the relevant values as a_gs
ii += BLOCKSIZE * gridDim.x;
}
}