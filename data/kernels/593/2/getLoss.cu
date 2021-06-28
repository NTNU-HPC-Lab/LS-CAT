#include "includes.h"
__global__ void getLoss(float* dat, float* rst){
*rst = -logf(*dat);
}