#include "includes.h"
__global__ void SetElement(float *vector , int position , float what) {
vector[position] = what;
}