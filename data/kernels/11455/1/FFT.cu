#include "includes.h"
//**********************************
//Nathan Durst
//FFT Cuda Program
//December, 5 2016
//**********************************
//This application uses cuda c and implements
// the Cooley-Tukey FFT algorithm to transforms
// an array of complex numbers into a data set
// correlation of complex numbers.
#define N 16384
#define PI 3.14

//kernel function declaration

__global__ void FFT(float * R, float * I, float * xR, float * xI)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
float real = 0, imag = 0;

//iterate through entire array for each index and calculate even
// and odd for real and imaginary numbers.
for (int i = 0; i<(N/2); i++)
{
//even
real += R[i] * cos((2*PI*(i*2))/N) - I[i] * sin((2*PI*id*(i*2))/N);
imag += R[i] * -sin((2*PI*(i*2))/N) + I[i] * cos((2*PI*id*(i*2))/N);

//odd
real += R[i] * cos((2*PI*(i*2+1))/N) - I[i] * sin((2*PI*id*(i*2+1))/N);
imag += R[i] * -sin((2*PI*(i*2+1))/N) + I[i] * cos((2*PI*id*(i*2+1))/N);
}
xR[id] = real;
xI[id] = imag;
}