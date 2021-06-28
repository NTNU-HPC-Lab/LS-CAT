#include "includes.h"


__device__ double caculateValueOfWeight(double parameter, int sign, double alpha)
{
return (parameter*sign*alpha);
}
__global__ void updateWeights(double* weights, double* parameters,double* otherp, int sign, double alpha)
{
int index = threadIdx.x;
double value = weights[index];
weights[index] = value + caculateValueOfWeight( parameters[index], sign, alpha);

}