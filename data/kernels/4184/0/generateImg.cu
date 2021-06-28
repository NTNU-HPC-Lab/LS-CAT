#include "includes.h"
__global__ void generateImg(unsigned char * data, unsigned char * img, unsigned char * tabDepth, int4 * _tabParents, int i, int tailleTab) {
int thx = blockIdx.x * blockDim.x + threadIdx.x;
int thy = blockIdx.y * blockDim.y + threadIdx.y;
int ThId = thy * tailleTab + thx;
int nbPar = 0;

if(data[ThId] == 0 && tabDepth[ThId] == i  && i != 1) {

if(_tabParents[ThId].x != -1) nbPar ++;
if(_tabParents[ThId].y != -1) nbPar ++;
if(_tabParents[ThId].z != -1) nbPar ++;
if(_tabParents[ThId].w != -1) nbPar ++;

data[ThId] = (data[_tabParents[ThId].x] + data[_tabParents[ThId].y] + data[_tabParents[ThId].z] + data[_tabParents[ThId].w]) / nbPar;

img[ThId] = data[ThId];
}

}