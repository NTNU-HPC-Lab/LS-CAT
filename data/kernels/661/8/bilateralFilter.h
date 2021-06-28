#include <stdio.h>
#include <cuda.h>
#include <vector>
#include "cuda_runtime.h"
#include "opencv2/highgui/highgui.hpp"
#include "timer.h"

using namespace std;
using namespace cv;

inline float3 multiply(const float a, const float3 b) {
	return{ a * b.x, a * b.y, a * b.z };
}

inline float3 add(const float3 a, const float3 b) {
	return{ a.x + b.x, a.y + b.y, a.z + b.z };
}

namespace bilateralFilter {
	Mat cpufilter(Mat im, float l2norm, int radius);
	Mat gpufilter(Mat im, float l2norm, int radius);
}