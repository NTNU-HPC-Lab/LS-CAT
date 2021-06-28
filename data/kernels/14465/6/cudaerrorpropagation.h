#ifndef CUDA_ERROR_PROPAGATION_H
#define CUDA_ERROR_PROPAGATION_H

#define CUDALUB_EXPORTS

#ifdef CUDALUB_EXPORTS
#define CULIBAPI __declspec(dllexport)
#else
#define CULIBAPI __declspec(dllimport)
#endif

typedef float(*func_ptr)(float x);

enum ActivationFuncType
{
	UNIPOLAR_SIGMOID = 0,
	BIPOLAR_SIGMOID = 1,
	SINUSOID = 2,
	LINEAR = 3
};

enum ErrorPropagationType
{
	BACK_PROPAGATION = 0,
	RESILIENT_PROPAGATION = 1
};

struct CudaErrorPropagation
{
	// Network and data
	int numInput;
	int numHidden;
	int numOutput;
	int numSamples;

	float *d_inputsBatch /*2d*/;
	float *d_inputHiddenWeights /*2d*/;
	float *d_hiddenOutputsBatch /*2d*/;
	float *d_hiddenOutputWeights /*2d*/;
	float *d_outputsBatch /*2d*/;

	// Propagation
	float *d_targetOutputsBatch /*2d*/;
	float *d_outputDeltasBatch /*2d*/;
	float *d_hiddenOutputGradients /*2d*/;
	float *d_hiddenDeltasBatch /*2d*/;
	float *d_inputHiddenGradients /*2d*/;
	float *d_errorsOutputsBatch /*2d*/;
	float *d_errorsBatch;
	float *d_error; /* Single value */

	float *d_hiddenOutputGradientsBatch /*3d*/;
	float *d_inputHiddenGradientsBatch /*3d*/;

	// BackPropagation
	float *d_previousInputHiddenWeightDeltas /*2d*/;
	float *d_previousHiddenOutputWeightDeltas /*2d*/;

	// ResilientPropagation
	float *d_previousInputHiddenGradients /*2d*/;
	float *d_previousHiddenOutputGradients /*2d*/;
	float *d_inputHiddenLearningRates /*2d*/;
	float *d_hiddenOutputLearningRates /*2d*/;

	// Computed weights
	float *h_inputHiddenWeights /*2d*/;
	float *h_hiddenOutputWeights /*2d*/;

	func_ptr h_pHiddenFunction;
	func_ptr h_pHiddenDerivative;
	func_ptr h_pOutputFunction;
	func_ptr h_pOutputDerivative;
};

#ifdef __cplusplus
extern "C" {
#endif

	CULIBAPI bool checkCudaSupport();

	CULIBAPI CudaErrorPropagation* createErrorPropagation(float *h_inputData /*2d*/, float *h_outputData /*2d*/,
		float *h_inputHiddenWeights /*2d*/, float *h_hiddenOutputWeights /*2d*/,
		int numInput, int numHidden, int numOutput, int numSamples,
		ActivationFuncType hiddenFunc, ActivationFuncType outputFunc);

	CULIBAPI void destroyErrorPropagation(CudaErrorPropagation *propagation);

	CULIBAPI const float* getInputHiddenWeights(CudaErrorPropagation *propagation);

	CULIBAPI const float* getHiddenOutputWeights(CudaErrorPropagation *propagation);

	CULIBAPI void randomizeWeights(CudaErrorPropagation *propagation);

	CULIBAPI float performBackPropEpoch(CudaErrorPropagation *propagation, float learningRate, float momentum);

	CULIBAPI float performResilientPropEpoch(CudaErrorPropagation *propagation);

#ifdef __cplusplus
}
#endif

#endif // CUDA_ERROR_PROPAGATION_H