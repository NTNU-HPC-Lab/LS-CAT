#ifndef ALGORITHM_CUH
#define ALGORITHM_CUH

#include "utils/scimage.h"

void ColorSpaceConvertion(ScGPUImage *src, ScGPUImage *dst);

void HorizontalReversalRGBA(ScGPUImage *src, ScGPUImage *dst);

void VerticalReversalRGBA(ScGPUImage *src, ScGPUImage *dst);

void RenderRGBAImageToSurface(ScGPUImage *image, cudaSurfaceObject_t surface);



#endif // ALGORITHM_CUH
