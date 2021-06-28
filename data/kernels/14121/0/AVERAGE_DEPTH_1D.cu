#include "includes.h"

// Kernel Average with Depth
extern "C"

//Converting 2D coordinates into one 1D coordinate
__global__ void AVERAGE_DEPTH_1D(int envSizeX, int envSizeY, float* envData, int depth){
int tidX = blockIdx.x * blockDim.x + threadIdx.x;
int tidY = blockIdx.y * blockDim.y + threadIdx.y;

float moyenne = 0;
int nbNombre = 0;

if(tidX < envSizeX && tidY < envSizeY){
for(int l = tidX - depth; l <= tidX + depth; l++){
if(l < 0){
int ltemp = l;
ltemp += envSizeX;

for(int k = tidY - depth; k <= tidY + depth; k++){
if(k < 0){
int ktemp = k;
ktemp += envSizeY;
if(envData[envSizeX * ltemp + ktemp] != -1){
moyenne += envData[envSizeX * ltemp + ktemp];
nbNombre++;
}
}
else if(k > envSizeY - 1){
int ktemp = k;
ktemp -= envSizeY;
if(envData[envSizeX * ltemp + ktemp] != -1){
moyenne += envData[envSizeX * ltemp + ktemp];
nbNombre++;
}
}
else{
if(envData[envSizeX * ltemp + k] != -1){
moyenne += envData[envSizeX * ltemp + k];
nbNombre++;
}
}
}
}
else if(l > envSizeX - 1){
int ltemp = l;
ltemp -= envSizeX;

for(int k = tidY - depth; k <= tidY + depth; k++){
if(k < 0){
int ktemp = k;
ktemp += envSizeY;
if(envData[envSizeX * ltemp + ktemp] != -1){
moyenne += envData[envSizeX * ltemp + ktemp];
nbNombre++;
}
}
else if(k > envSizeY - 1){
int ktemp = k;
ktemp -= envSizeY;
if(envData[envSizeX * ltemp + ktemp] != -1){
moyenne += envData[envSizeX * ltemp + ktemp];
nbNombre++;
}
}
else{
if(envData[envSizeX * ltemp + k] != -1){
moyenne += envData[envSizeX * ltemp + k];
nbNombre++;
}
}
}
}
else{
for(int k = tidY - depth; k <= tidY + depth; k++){
if(k < 0){
int ktemp = k;
ktemp += envSizeY;
if(envData[envSizeX * l + ktemp] != -1){
moyenne += envData[envSizeX * l + ktemp];
nbNombre++;
}
}
else if(k > envSizeY - 1){
int ktemp = k;
ktemp -= envSizeY;
if(envData[envSizeX * l + ktemp] != -1){
moyenne += envData[envSizeX * l + ktemp];
nbNombre++;
}
}
else{
if(envData[envSizeX * l + k] != -1){
moyenne += envData[envSizeX * l + k];
nbNombre++;
}
}
}
}
}
if(nbNombre != 0){
envData[envSizeX * tidX + tidY] = moyenne / nbNombre;
}
}
__syncthreads();
}