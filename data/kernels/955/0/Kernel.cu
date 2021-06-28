#include "includes.h"



#define TIME                5.
#define TIME_STEP           .1

#define STEP                1.
#define K                   TIME_STEP / SQUARE(STEP)

#define SQUARE(x)           (x * x)
#define HANDLE_ERROR(err)   (HandleError(err, __FILE__, __LINE__))


__global__ void Kernel(double * device, const uint size)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;

if (i == 0) {
device[i] = .0;
} else if (i == size - 1) {
device[size - 1] = device[size - 2] + 5 * STEP;
} else if (i < size) {
device[i] = (device[i + 1] - 2 * device[i] + device[i - 1]) * K + device[i];
}
}