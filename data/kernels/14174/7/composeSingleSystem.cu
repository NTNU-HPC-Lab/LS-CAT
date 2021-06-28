#include "includes.h"
__device__ int roundToInt(float val)
{
return (int)floor(val + 0.5f);
}
__device__ float d_priorF;  __global__ void add(float *p, float *q) { *p += *q; }
__global__ void composeSingleSystem(const size_t offset, const float *H, const size_t lowresWidth,  const size_t lowresHeight, const size_t highresWidth, const size_t highresHeight, const float psfWidth, const int pixelRadius, float *systemMatrixVals, int *systemMatrixCols, int *systemMatrixRows)
{
const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

const size_t lowresPixels  = lowresWidth  * lowresHeight;

if (idx >= lowresPixels)
return;

// Coordinates of this thread in the low-res image
size_t x = idx % lowresWidth;
size_t y = idx / lowresWidth;

// Row that this thread writes in the full system matrix
size_t r = idx + offset;

// Transform pixel coordinates from the LR grid to the desired HR grid

float hrx, hry;
float zoom = float(highresWidth) / float(lowresWidth);

hrx = (H[0] * x + H[1] * y + H[2]) * zoom;
hry = (H[3] * x + H[4] * y + H[5]) * zoom;

float weightSum = 0.0f;

const size_t maxRowElems = (2 * pixelRadius + 1) * (2 * pixelRadius + 1);
size_t offsetCRS = 0;
size_t offsetRows = maxRowElems * r;

// Iterate over the neighborhood defined by the width of the psf
for (int offsetY = -pixelRadius; offsetY <= pixelRadius; ++offsetY)
{
const int ny = roundToInt(hry + offsetY);

if (ny < 0 || ny >= highresHeight)
continue;

for (int offsetX = -pixelRadius; offsetX <= pixelRadius; ++offsetX)
{
const int nx = roundToInt(hrx + offsetX);

if (nx < 0 || nx >= highresWidth)
continue;

const float dx = hrx - float(nx);
const float dy = hry - float(ny);

// Compute influence of current high-res pixel for
// this thread's low-res pixel

float dist = dx*dx*H[0]*H[0] + dy*dy*H[4]*H[4] +
dx*dy*H[0]*H[3] + dx*dy*H[1]*H[4];

float weight = expf(-dist / (2.0f * zoom * zoom * psfWidth * psfWidth));

const size_t valIdx = offsetRows + offsetCRS;
systemMatrixVals[valIdx] = weight;
systemMatrixCols[valIdx] = ny * highresWidth + nx;

weightSum += weight;

++offsetCRS;
}
}

if (weightSum > 0.0f)
{
// Normalize row sums
for (size_t i = 0; i < offsetCRS; ++i)
{
systemMatrixVals[offsetRows + i] /= weightSum;
}
}

// If we have saved less than maxRowElems elements,
// we have to pad the CRS structure with 0 entries
// to make sure it is valid

if (offsetCRS == 0)
{
systemMatrixVals[offsetRows] = 0.0f;
systemMatrixCols[offsetRows] = 0;
++offsetCRS;
}

bool copy = false;

// Try adding elements after the last saved entry

while (offsetCRS < maxRowElems)
{
const size_t idx = offsetRows + offsetCRS;

if (systemMatrixCols[idx - 1] + 1 >= highresWidth * highresHeight)
{
copy = true;
break;
}

systemMatrixVals[idx] = 0.0f;
systemMatrixCols[idx] = systemMatrixCols[idx - 1] + 1;
offsetCRS++;
}

// If there isn't enough space after the last saved
// entry, add padding before first entry

if (copy)
{
for (int idx = offsetCRS - 1; idx >= 0; --idx)
{
systemMatrixVals[offsetRows + maxRowElems - (offsetCRS - idx)] =
systemMatrixVals[offsetRows + idx];
systemMatrixCols[offsetRows + maxRowElems - (offsetCRS - idx)] =
systemMatrixCols[offsetRows + idx];
}

for (int idx = maxRowElems - offsetCRS - 1; idx >= 0; --idx)
{
systemMatrixVals[offsetRows + idx] = 0.0f;
systemMatrixCols[offsetRows + idx] = systemMatrixCols[offsetRows + idx + 1] - 1;
}
}

systemMatrixRows[r] = r * maxRowElems;
}