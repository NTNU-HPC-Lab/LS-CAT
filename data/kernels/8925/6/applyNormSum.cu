#include "includes.h"
__global__ void applyNormSum(double *dMap,double *dSupFeature, double *dMaxSupFeature, double *dMeanSupFeature, double *dInfFeature, double *dMaxInfFeature, double *dMeanInfFeature, int dSize){
int tid = threadIdx.x + blockIdx.x * blockDim.x;

double SupCoeff = (dMaxSupFeature[0] - dMeanSupFeature[0])*(dMaxSupFeature[0] - dMeanSupFeature[0]);
double InfCoeff = (dMaxInfFeature[0] - dMeanInfFeature[0])*(dMaxInfFeature[0] - dMeanInfFeature[0]);

while (tid < dSize) {
dMap[tid] += dSupFeature[tid]*SupCoeff + dInfFeature[tid]*InfCoeff;
tid  += blockDim.x * gridDim.x;
}
}