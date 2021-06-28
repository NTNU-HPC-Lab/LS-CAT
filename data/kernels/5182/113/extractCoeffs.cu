#include "includes.h"
__global__ void extractCoeffs(const float  *matrices, float *coeffArray, int savedCoeffs, int dimsize){
int threadGlobalID = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

int offsetMatrix = threadGlobalID * dimsize * dimsize,
offsetCoeff = threadGlobalID * savedCoeffs,
coeffsLeft = savedCoeffs,
x, y, y_n = 0, x_n = 1,
numberinrow, tmp;

coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)] = matrices[offsetMatrix];
coeffsLeft -= 1;

while (coeffsLeft > 0){
// Work out number in row
x = x_n;
y = y_n;

if (x_n < dimsize - 1)
numberinrow = x_n + 1;
else
numberinrow = x_n - (y_n - 1);

if (numberinrow % 2 == 0){
// Even
while (numberinrow > 0 && coeffsLeft > 0){
coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)] = matrices[offsetMatrix + x + y * dimsize];
numberinrow--;
coeffsLeft--;

if ((numberinrow + 1) % 2 == 0){
// Swap x and y
tmp = x;
x = y;
y = tmp;
}
else{
// Swap x and y
tmp = x;
x = y;
y = tmp;
x--;
y++;
}
}
}
else{
// Odd
while (numberinrow > 1 && coeffsLeft > 0){
coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)] = matrices[offsetMatrix + x + y * dimsize];
numberinrow--;
coeffsLeft--;
if ((numberinrow + 1) % 2 == 1){
// Swap x and y
tmp = x;
x = y;
y = tmp;
}
else{
// Swap x and y
tmp = x;
x = y;
y = tmp;
x--;
y++;
}
}
if (coeffsLeft > 0){
// add the odd one
coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)] = matrices[offsetMatrix + x + y * dimsize];
numberinrow--;
coeffsLeft--;
}
}
if (x_n == dimsize - 1){
y_n++;
}
else{
x_n++;
}
}
}