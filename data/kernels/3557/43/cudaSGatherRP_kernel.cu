#include "includes.h"
__global__ void cudaSGatherRP_kernel( unsigned int inputSizeX, unsigned int inputSizeY, unsigned int nbAnchors, unsigned int batchSize, const float* inputs, const int* i, const int* j, const int* k, const int* b, const int* mask, float* outputs, int* anchors, unsigned int topN, const unsigned int nbProposals)
{
const int batchPos = blockIdx.z;
const int sortOffset = batchPos*topN;
const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;
const int totalIndex = index + sortOffset;
const int batchIndex = index + batchPos*nbProposals;

if(index < nbProposals)
{
unsigned int xIdx = i[ mask[totalIndex] + sortOffset ]
+ j[ mask[totalIndex] + sortOffset ]*inputSizeX
+ (k[ mask[totalIndex] + sortOffset ] + nbAnchors)*inputSizeX*inputSizeY
+ b[ mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

unsigned int yIdx = i[ mask[totalIndex] + sortOffset ]
+ j[ mask[totalIndex] + sortOffset ]*inputSizeX
+ (k[ mask[totalIndex] + sortOffset ] + 2*nbAnchors)*inputSizeX*inputSizeY
+ b[ mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

unsigned int wIdx = i[ mask[totalIndex] + sortOffset ]
+ j[ mask[totalIndex] + sortOffset ]*inputSizeX
+ (k[ mask[totalIndex] + sortOffset ] + 3*nbAnchors)*inputSizeX*inputSizeY
+ b[ mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

unsigned int hIdx = i[ mask[totalIndex] + sortOffset ]
+ j[ mask[totalIndex] + sortOffset ]*inputSizeX
+ (k[ mask[totalIndex] + sortOffset ] + 4*nbAnchors)*inputSizeX*inputSizeY
+ b[ mask[totalIndex] + sortOffset ]*nbAnchors*inputSizeX*inputSizeY*6;

anchors[0 + (batchIndex)*4] = i[mask[totalIndex]];
anchors[1 + (batchIndex)*4] = j[mask[totalIndex]];
anchors[2 + (batchIndex)*4] = k[mask[totalIndex]];
anchors[3 + (batchIndex)*4] = b[mask[totalIndex]];

outputs[0 + (batchIndex)*4] = inputs[xIdx];
outputs[1 + (batchIndex)*4] = inputs[yIdx];
outputs[2 + (batchIndex)*4] = inputs[wIdx];
outputs[3 + (batchIndex)*4] = inputs[hIdx];
}

}