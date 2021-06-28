#include "includes.h"
__global__ void cudaS_ssdToOutput_kernels(  unsigned int batchSize, unsigned int nbClass, unsigned int nbAnchors, unsigned int channelWidth, unsigned int channelHeight, unsigned int nbProposals, unsigned int* nbValidROIs, unsigned int cls, unsigned int totalParts, unsigned int totalTemplates, unsigned int maxParts, unsigned int maxTemplates, unsigned int cumulParts, unsigned int cumulTemplates, unsigned int nbParts, unsigned int nbTemplates, float xRatio, float yRatio, float xOutputRatio, float yOutputRatio, const float* roi_bbox, const float* roi_anchors, const float* anchors, const float* inputs_parts, const float* inputs_templates, float* outputs)
{
const int batchPos = blockIdx.z;
const int proposal = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;
const int ptIdx = blockIdx.y;
const unsigned int nbAnchorPerCls = nbAnchors;

const int nbDetectedObject  = (int) nbValidROIs[batchPos];
const int nbIdx = 6;
if(proposal < nbProposals)
{
const unsigned int n = proposal + cls*nbProposals + batchPos*nbProposals*nbClass;

if(proposal < nbDetectedObject)
{
if(ptIdx == 0)
{
outputs[0 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[0 + 5*proposal + batchPos*nbProposals*5];
outputs[1 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[1 + 5*proposal + batchPos*nbProposals*5];
outputs[2 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[2 + 5*proposal + batchPos*nbProposals*5];
outputs[3 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[3 + 5*proposal + batchPos*nbProposals*5];
outputs[4 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = roi_bbox[4 + 5*proposal + batchPos*nbProposals*5];
outputs[5 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = (float) cls;
}

if(ptIdx < nbParts && totalParts > 0)
{
const unsigned int xa   = roi_anchors[0 + 5*proposal + batchPos*nbProposals*5];
const unsigned int ya   = roi_anchors[1 + 5*proposal + batchPos*nbProposals*5];
const unsigned int k    = roi_anchors[2 + 5*proposal + batchPos*nbProposals*5];


const int yIdx = xa
+ ya*channelWidth
+ (k*nbParts*2 + cumulParts + ptIdx*2)*channelHeight*channelWidth
+ batchPos*channelHeight*channelWidth*nbAnchorPerCls*2*totalParts;
const int xIdx = xa
+ ya*channelWidth
+ (k*nbParts*2 + cumulParts + ptIdx*2 + 1)*channelHeight*channelWidth
+ batchPos*channelHeight*channelWidth*nbAnchorPerCls*2*totalParts;


const float partY = inputs_parts[yIdx];
const float partX = inputs_parts[xIdx];

const int xa0 = (int)(anchors[cls*4*nbAnchorPerCls + k*4] + xa * xRatio);
const int ya0 = (int)(anchors[cls*4*nbAnchorPerCls + k*4 + 1] + ya * yRatio);
const int xa1 = (int)(anchors[cls*4*nbAnchorPerCls + k*4 + 2] + xa * xRatio);
const int ya1 = (int)(anchors[cls*4*nbAnchorPerCls + k*4 + 3] + ya * yRatio);

// Anchors width and height
const int wa = xa1 - xa0;
const int ha = ya1 - ya0;

// Anchor center coordinates (xac, yac)
const float xac = xa0 + wa / 2.0;
const float yac = ya0 + ha / 2.0;
const float predPartY = ((partY) * ha + yac)*yOutputRatio ;
const float predPartX = ((partX) * wa + xac)*xOutputRatio ;

outputs[ptIdx*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = predPartY;
outputs[ptIdx*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = predPartX;

}
else if(ptIdx < maxParts && totalParts > 0)
{
outputs[ptIdx*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
outputs[ptIdx*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
}

///for(unsigned int t = 0; t < nbTemplates; ++t)
if(ptIdx < nbTemplates && totalTemplates > 0)
{
const unsigned int xa   = roi_anchors[0 + 5*proposal + batchPos*nbProposals*5];
const unsigned int ya   = roi_anchors[1 + 5*proposal + batchPos*nbProposals*5];
const unsigned int k    = roi_anchors[2 + 5*proposal + batchPos*nbProposals*5];

const int yIdx = xa
+ ya*channelWidth
+ (k*nbTemplates*3 + cumulTemplates + ptIdx*3)*channelHeight*channelWidth
+ batchPos*channelHeight*channelWidth*nbAnchorPerCls*3*totalTemplates;
const int xIdx = xa
+ ya*channelWidth
+ (k*nbTemplates*3 + cumulTemplates + ptIdx*3 + 1)*channelHeight*channelWidth
+ batchPos*channelHeight*channelWidth*nbAnchorPerCls*3*totalTemplates;
const int zIdx = xa
+ ya*channelWidth
+ (k*nbTemplates*3 + cumulTemplates + ptIdx*3 + 2)*channelHeight*channelWidth
+ batchPos*channelHeight*channelWidth*nbAnchorPerCls*3*totalTemplates;


const float templateY = expf(inputs_templates[yIdx]);
const float templateX = expf(inputs_templates[xIdx]);
const float templateZ = expf(inputs_templates[zIdx]);

outputs[ptIdx*3 + maxParts*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = templateY;
outputs[ptIdx*3 + maxParts*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = templateX;
outputs[ptIdx*3 + maxParts*2 + 2 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = templateZ;

}
else if(ptIdx < maxTemplates && totalTemplates > 0)
{
outputs[ptIdx*3 + maxParts*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
outputs[ptIdx*3 + maxParts*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
outputs[ptIdx*3 + maxParts*2 + 2 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
}

}
else
{
outputs[0 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
outputs[1 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
outputs[2 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
outputs[3 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;
outputs[4 + n*(nbIdx + maxParts*2 + maxTemplates*3)] = 0.0;

//for(unsigned int p = 0; p < nbParts; ++p)
if(ptIdx < maxParts && totalParts > 0)
{
outputs[ptIdx*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
outputs[ptIdx*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
}

//for(unsigned int t = 0;t < nbTemplates; ++t)
if(ptIdx < maxTemplates && totalTemplates > 0)
{
outputs[ptIdx*3 + maxParts*2 + 0 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
outputs[ptIdx*3 + maxParts*2 + 1 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
outputs[ptIdx*3 + maxParts*2 + 2 + nbIdx + n*(nbIdx + maxParts*2 + maxTemplates*3) ] = 0.0;
}

}
}
}