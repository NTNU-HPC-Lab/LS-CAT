#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void randControls(int noPaths, int nYears, int noControls, float* randCont, int* control) {
int idx = blockIdx.x*blockDim.x + threadIdx.x;

if (idx < noPaths*nYears) {
control[idx] = (int)(randCont[idx]*noControls);
if (control[idx] == noControls) {
control[idx]--;
}
}
}