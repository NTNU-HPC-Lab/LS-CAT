#include "includes.h"
__global__ void call_kALACSearch(int16_t * mCoefsU, int16_t * mCoefsV, int32_t kALACMaxCoefs)
{
int x = blockIdx.x;
int y = threadIdx.x;

int index = x * 16 * 16 + y * 16;
int32_t		k;
int32_t		den = 1 << DENSHIFT_DEFAULT;

mCoefsU[index + 0] = (AINIT * den) >> 4;
mCoefsU[index + 1] = (BINIT * den) >> 4;
mCoefsU[index + 2] = (CINIT * den) >> 4;

mCoefsV[index + 0] = (AINIT * den) >> 4;
mCoefsV[index + 1] = (BINIT * den) >> 4;
mCoefsV[index + 2] = (CINIT * den) >> 4;

for (k = 3; k < kALACMaxCoefs; k++)
{
mCoefsU[index + k] = 0;
mCoefsV[index + k] = 0;
}
}