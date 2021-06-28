#include "includes.h"

// ERROR CHECKING MACROS //////////////////////////////////////////////////////

__global__ void printControls(int noPaths, int path, int nYears, int* controls) {
for (int ii = 0; ii < nYears; ii++) {
printf("%d %d\n",ii,controls[path*nYears + ii]);
}
}