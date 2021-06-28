/*
 * HiddenLayer.h
 *
 *  Created on: Nov 28, 2015
 *      Author: tdx
 */

#ifndef HIDDENLAYER_H_
#define HIDDENLAYER_H_

#include"LayersBase.h"
#include"../common/cuMatrix.h"
#include"../common/cuMatrixVector.h"
#include"../common/utility.cuh"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"
#include<cuda_runtime.h>
#include<math.h>
#include "curand.h"

/*
 * Class Hidden layer
 * */
class HiddenLayer: public LayersBase
{
public:
	HiddenLayer(string name, int sign);
	HiddenLayer(const HiddenLayer* layer);
    HiddenLayer(const configBase* templateConfig);
	~HiddenLayer();
	void initRandom();
    void ReShape(){}
	void forwardPropagation(string train_or_test);
	void backwardPropagation(float Momentum);
	void saveWeight(FILE*file);
	void readWeight(FILE*file);
	void dropOut(float*data, int size, float dropout_rate);
	void createHandles();
	void destroyHandles();
    int getOutputSize();
    void compute_cost();

private:

    curandGenerator_t curandGenerator_W;
    curandGenerator_t curandGenerator_B;
	float* dev_Weight, *host_Weight;
	float* dev_Bias, *host_Bias;
	float* dev_Wgrad,*dev_Bgrad;
	float*tmp_Wgrad, *tmp_Bgrad;
    float* dev_weightSquare;
    float* host_weightSquare;
	float epsilon;
	float* VectorOnes;
	int inputSize;
	int outputSize;
	int batchSize;
	int prev_num;
	int prev_channels;
	int prev_height;
	int prev_width;
	float lambda;
};



#endif /* HIDDENLAYER_H_ */
