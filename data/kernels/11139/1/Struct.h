#pragma once
#include <vector>

struct HostImage
{
	unsigned char* data;
	int cols;
	int rows;
};

struct DeviceImage
{
	unsigned char* inputImage;
	unsigned char* outputImage;
	float* hue;
	float* saturation;
	unsigned char* value;
	unsigned char* valueBlurred;
	unsigned char* valueContrast;
	unsigned char* mask;
	float* filter;
	int filterWidth;
	int width;
	int height;
};

struct GaussianFilter
{
	std::vector<float> weight;				// array of weights
	int			  width;				// filter width
};