#include "includes.h"
__global__ void cudaSToOutput_kernel( const unsigned int nbProposals, const unsigned int scoreIdx, const unsigned int nbCls, const unsigned int nbOutputs, const unsigned int maxParts, const unsigned int maxTemplates, bool generateParts, bool generateTemplates, const int* numPartsPerClass, const int* numTemplatesPerClass, const int* maxCls, const float* ROIEst, const int* predictionIndex, const float* partsPrediction, const float* partsVisibilityPrediction, const float* templatesPrediction, float* outputs)
{
const int batchPos = blockIdx.z*nbProposals;
const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

if(index < nbProposals)
{
const unsigned int inputIdx = index*4*(nbCls - scoreIdx)
+ batchPos*4*(nbCls - scoreIdx);
unsigned int outputIdx = 0;
unsigned offset = 0;

if((nbOutputs == 4))
outputIdx = index*4 + batchPos*4;
else if((nbOutputs == 5))
outputIdx = index*5 + batchPos*5;
else if(generateParts && generateTemplates)
outputIdx = (index + batchPos)*(5 + maxParts*3 + maxTemplates*3);
else if(generateTemplates)
outputIdx = (index + batchPos)*(5 + maxTemplates*3);
else if(generateParts)
outputIdx = (index + batchPos)*(5 + maxParts*3);

outputs[0 + outputIdx] = ROIEst[0 + inputIdx];
outputs[1 + outputIdx] = ROIEst[1 + inputIdx];
outputs[2 + outputIdx] = ROIEst[2 + inputIdx];
outputs[3 + outputIdx] = ROIEst[3 + inputIdx];

offset += 4;

if(nbOutputs > 4)
{
int cls = maxCls[index + batchPos];
outputs[4 + outputIdx] = cls > -1 ?
(float) cls
: 0.0;
offset += 1;
}

if(generateParts)
{
const int predProp = predictionIndex[(index + batchPos)*2 + 0];
const int predCls = predictionIndex[(index + batchPos)*2 + 1];

if(predCls > -1)
{
// PARTS PROCESSING
for(unsigned int part = 0; part < numPartsPerClass[predCls];
++part)
{
const unsigned int partIdx = batchPos*maxParts*2*nbCls
+ predProp*maxParts*2*nbCls
+ predCls*maxParts*2
+ part*2;
outputs[0 + offset + part*2 + outputIdx] = partsPrediction[0 + partIdx];
outputs[1 + offset + part*2 + outputIdx] = partsPrediction[1 + partIdx];

}
for(int idx = numPartsPerClass[predCls]; idx < maxParts; ++idx)
{
outputs[0 + offset + numPartsPerClass[predCls]*2 + idx*2 + outputIdx] = 0.0;
outputs[1 + offset + numPartsPerClass[predCls]*2 + idx*2 + outputIdx] = 0.0;
}
}

offset += 2*maxParts;

if(predCls > -1)
{
// PARTS VISIBILITY PROCESSING
for(unsigned int part = 0; part < numPartsPerClass[predCls];
++part)
{
const unsigned int partVisibilityIdx = batchPos*maxParts*nbCls
+ predProp*maxParts*nbCls
+ predCls*maxParts
+ part;
outputs[offset + part + outputIdx] = partsVisibilityPrediction[partVisibilityIdx];

}

for(int idx = numPartsPerClass[predCls]; idx < maxParts; ++idx)
outputs[offset + numPartsPerClass[predCls] + idx + outputIdx] = -1.0;
}
offset += maxParts;
}

if(generateTemplates)
{

const int predProp = predictionIndex[(index + batchPos)*2 + 0];
const int predCls = predictionIndex[(index + batchPos)*2 + 1];

if(predCls > -1)
{
for(unsigned int tpl = 0; tpl < numTemplatesPerClass[predCls]; ++tpl)
{
unsigned int templateIdx = batchPos*maxTemplates*3*nbCls
+ predProp*maxTemplates*3*nbCls
+ predCls*maxTemplates*3
+ tpl*3;

outputs[0 + offset + tpl*3 + outputIdx] = templatesPrediction[0 + templateIdx];
outputs[1 + offset + tpl*3 + outputIdx] = templatesPrediction[1 + templateIdx];
outputs[2 + offset + tpl*3 + outputIdx] = templatesPrediction[2 + templateIdx];

}
for(int idx = numTemplatesPerClass[predCls]; idx < maxParts; ++idx)
{
outputs[0 + offset + numTemplatesPerClass[predCls]*3 + idx*3 + outputIdx] = 0.0;
outputs[1 + offset + numTemplatesPerClass[predCls]*3 + idx*3 + outputIdx] = 0.0;
outputs[2 + offset + numTemplatesPerClass[predCls]*3 + idx*3 + outputIdx] = 0.0;

}

}
}
}

}