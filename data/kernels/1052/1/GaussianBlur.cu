#include "includes.h"
__global__ void GaussianBlur(unsigned int *B,unsigned int *G,unsigned int *R, int numberOfPixels, unsigned int width, int *B_new, int *G_new, int *R_new)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index >= numberOfPixels){
//printf("%d\n",index);
return;
}

int mask[] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
int s = mask[0] + mask[1] + mask[2] + mask[3] + mask[4] + mask[5] + mask[6] + mask[7] + mask[8];

if (index < width){ // dolny rzad pikseli
if (index == 0){ //lewy dolny rog
s = mask[4] + mask[1] + mask[2] + mask[5];
B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width + 1] * mask[2] + B[index + 1] * mask[5]) / s);
G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width + 1] * mask[2] + G[index + 1] * mask[5]) / s);
R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width + 1] * mask[2] + R[index + 1] * mask[5]) / s);
return;
}

if (index == width - 1){//prawy dolny rog
s = mask[4] + mask[0] + mask[1] + mask[3];
B_new[index] = (B[index] * mask[4] + B[index + width - 1] * mask[0] + B[index + width] * mask[1] + B[index - 1] * mask[3]);
G_new[index] = (G[index] * mask[4] + G[index + width - 1] * mask[0] + G[index + width] * mask[1] + G[index - 1] * mask[3]);
R_new[index] = (R[index] * mask[4] + R[index + width - 1] * mask[0] + R[index + width] * mask[1] + R[index - 1] * mask[3]);
return;
}
//reszta pikseli w dolnym rzedzie
s = mask[4] + mask[1] + mask[2] + mask[5] + mask[0] + mask[3];
B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width + 1] * mask[2] + B[index + 1] * mask[5] + B[index + width - 1] * mask[0] + B[index - 1] * mask[3]) / s);
R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width + 1] * mask[2] + R[index + 1] * mask[5] + R[index + width - 1] * mask[0] + R[index - 1] * mask[3]) / s);
G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width + 1] * mask[2] + G[index + 1] * mask[5] + G[index + width - 1] * mask[0] + G[index - 1] * mask[3]) / s);

return;
}
if (index >= numberOfPixels - width){ //gorny rzad pikseli

if (index == numberOfPixels - width){ //lewy gorny rog
s = mask[4] + mask[5] + mask[7] + mask[8];
B_new[index] = (int)((B[index] * mask[4] + B[index + 1] * mask[5] + B[index - width] * mask[7] + B[index - width + 1] * mask[8]) / s);
G_new[index] = (int)((G[index] * mask[4] + G[index + 1] * mask[5] + G[index - width] * mask[7] + G[index - width + 1] * mask[8]) / s);
R_new[index] = (int)((R[index] * mask[4] + R[index + 1] * mask[5] + R[index - width] * mask[7] + R[index - width + 1] * mask[8]) / s);
return;
}

if (index == numberOfPixels - 1){ //prawy gorny rog
s = mask[4] + mask[3] + mask[6] + mask[7];
B_new[index] = (int)((B[index] * mask[4] + B[index - 1] * mask[3] + B[index - width - 1] * mask[6] + B[index - width] * mask[7]) / s);
G_new[index] = (int)((G[index] * mask[4] + G[index - 1] * mask[3] + G[index - width - 1] * mask[6] + G[index - width] * mask[7]) / s);
R_new[index] = (int)((R[index] * mask[4] + R[index - 1] * mask[3] + R[index - width - 1] * mask[6] + R[index - width] * mask[7]) / s);
return;
}

s = mask[4] + mask[3] + mask[5] + mask[6] + mask[7] + mask[8];
B_new[index] = (int)((B[index] * mask[4] + B[index - 1] * mask[3] + B[index - width - 1] * mask[6] + B[index - width] * mask[7] + B[index + 1] * mask[5] + B[index - width] * mask[8]) / s);
R_new[index] = (int)((R[index] * mask[4] + R[index - 1] * mask[3] + R[index - width - 1] * mask[6] + R[index - width] * mask[7] + R[index + 1] * mask[5] + R[index - width] * mask[8]) / s);
G_new[index] = (int)((G[index] * mask[4] + G[index - 1] * mask[3] + G[index - width - 1] * mask[6] + G[index - width] * mask[7] + G[index + 1] * mask[5] + G[index - width] * mask[8]) / s);
return;
}
if (index % width == 0){ //lewa sciana
s = mask[4] + mask[1] + mask[2] + mask[5] + mask[8] + mask[7];
B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width + 1] * mask[2] + B[index + 1] * mask[5] + B[index - width + 1] * mask[8] + B[index - width]) / s);
G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width + 1] * mask[2] + G[index + 1] * mask[5] + G[index - width + 1] * mask[8] + G[index - width]) / s);
R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width + 1] * mask[2] + R[index + 1] * mask[5] + R[index - width + 1] * mask[8] + R[index - width]) / s);
return;
}
if (index % width == width - 1){ //prawa sciana
s = mask[4] + mask[1] + mask[0] + mask[3] + mask[6] + mask[7];
B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width - 1] * mask[0] + B[index - 1] * mask[3] + B[index - width - 1] * mask[6] + B[index - width] * mask[7]) / s);
R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width - 1] * mask[0] + R[index - 1] * mask[3] + R[index - width - 1] * mask[6] + R[index - width] * mask[7]) / s);
G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width - 1] * mask[0] + G[index - 1] * mask[3] + G[index - width - 1] * mask[6] + G[index - width] * mask[7]) / s);
return;
}


int poz_1 = index - width - 1;
int poz_2 = index - width;
int poz_3 = index - width + 1;
int poz_4 = index - 1;
int poz_5 = index;
int poz_6 = index + 1;
int poz_7 = index + width - 1;
int poz_8 = index + width;
int poz_9 = index + width + 1;

B_new[index] = (int)(((B[poz_1] * mask[0]) + (B[poz_2] * mask[1]) + (B[poz_3] * mask[2]) + (B[poz_4] * mask[3]) + (B[poz_5] * mask[4]) + (B[poz_6] * mask[5]) + (B[poz_7] * mask[6]) + (B[poz_8] * mask[7]) + (B[poz_9] * mask[8])) / s);
G_new[index] = (int)(((G[poz_1] * mask[0]) + (G[poz_2] * mask[1]) + (G[poz_3] * mask[2]) + (G[poz_4] * mask[3]) + (G[poz_5] * mask[4]) + (G[poz_6] * mask[5]) + (G[poz_7] * mask[6]) + (G[poz_8] * mask[7]) + (G[poz_9] * mask[8])) / s);
R_new[index] = (int)(((R[poz_1] * mask[0]) + (R[poz_2] * mask[1]) + (R[poz_3] * mask[2]) + (R[poz_4] * mask[3]) + (R[poz_5] * mask[4]) + (R[poz_6] * mask[5]) + (R[poz_7] * mask[6]) + (R[poz_8] * mask[7]) + (R[poz_9] * mask[8])) / s);


}