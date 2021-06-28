#include "includes.h"



// helper for CUDA error handling
__global__ void getSufficientComponentNum(const double* eigenvalues, std::size_t* componentNum, std::size_t eigenRows, double epsilon)
{
double variance = 0;
for(std::size_t i = 0; i < eigenRows; ++i)
{
variance += eigenvalues[i];
}
variance *= eigenRows;

(*componentNum) = 1;
double subVariance = eigenRows * eigenvalues[eigenRows - 1];
double explanatoryScore = subVariance / variance;
for(; (*componentNum) < eigenRows && explanatoryScore <= epsilon; (*componentNum) += 1)
{
subVariance += eigenRows * eigenvalues[eigenRows - (*componentNum) - 1];
explanatoryScore = subVariance / variance;
}
}