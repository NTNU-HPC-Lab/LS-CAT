#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#ifndef uint
#define uint unsigned int
#endif

#include "renderer.h"

class CudaRenderer : public Renderer
{

private:

	Image* image;
	SceneName sceneName;

	int numCircles;
	float* position;
	float* velocity;
	float* color;
	float* radius;

	float* cudaDevicePosition;
	float* cudaDeviceVelocity;
	float* cudaDeviceColor;
	float* cudaDeviceRadius;
	float* cudaDeviceImageData;

	unsigned int  m_binWidth;

	/*Used by method 0
	*/
	unsigned int* m_devicePredicate;
	unsigned int* m_deviceOffset;

	/*Used by method 1
	*/
	unsigned int* m_devCirBinCnt;
	unsigned int* m_devCirBinIdx;
	int* m_devBinStart;
	int* m_hstBinStart;

public:

	CudaRenderer();
	virtual ~CudaRenderer();

	const Image* getImage();

	void setup();

	void loadScene(SceneName name);

	void allocOutputImage(int width, int height);

	void clearImage();

	void advanceAnimation();

	void render();

	void shadePixel(
		int circleIndex,
		float pixelCenterX, float pixelCenterY,
		float px, float py, float pz,
		float* pixelData);
};

#endif