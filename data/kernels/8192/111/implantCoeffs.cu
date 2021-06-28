#include "includes.h"
__global__ void implantCoeffs(float* matrices, float *coeffArray, int savedCoeffs, int dimsize){

int id = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

int offsetMatrix = id * dimsize * dimsize,
offsetCoeff = id * savedCoeffs,
coeffsLeft = savedCoeffs,
x, y, y_n = 0, x_n = 1,
numberinrow, tmp;

matrices[offsetMatrix] = coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)];
coeffsLeft -= 1;

while (coeffsLeft > 0){
// Work out number in row
x = x_n;
y = y_n;

if (x_n < dimsize - 1){
numberinrow = x_n + 1;
}
else{
numberinrow = x_n - (y_n - 1);
}

if (numberinrow % 2 == 0){
// Even
while (numberinrow > 0 && coeffsLeft > 0){
matrices[offsetMatrix + x + y * dimsize] = coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)];
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
matrices[offsetMatrix + x + y * dimsize] = coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)];
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
matrices[offsetMatrix + x + y * dimsize] = coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)];
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