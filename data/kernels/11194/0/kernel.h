#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "Breakout.h"

void kernel_wrapper(int CIRCLE_SEGMENTS, float *xx, float*yy);