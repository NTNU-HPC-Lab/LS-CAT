/*
ref : ImageNet Classification with Deep Convolutional Neural Networks
*/
#ifndef __LAYERS_BRAHCH_LAYER_H__
#define __LAYERS_BRAHCH_LAYER_H__

#include "LayerBase.h"
#include "../common/cuMatrix.h"
#include <vector>
#include <map>
#include "../common/util.h"


class BranchLayer: public ConvLayerBase{
public:
	BranchLayer(std::string name);

	void feedforward();
	void backpropagation();
	void getGrad(){};
	void updateWeight(){};
	void clearMomentum(){};

	void calCost(){};
	void initFromCheckpoint(FILE* file){};
	void save(FILE* file){};

	~BranchLayer(){

	}

	cuMatrix<float>* getOutputs(){return NULL;}
	cuMatrix<float>* getCurDelta(){return NULL;}

	cuMatrix<float>* getSubOutput(std::string name){
		if(mapId.find(name) != mapId.end()){
			return outputs[mapId[name]];
		}else{
			printf("BranchLayer: can not find getOutputs %s\n", name.c_str());
			exit(0);
			return NULL;
		}
	};

	cuMatrix<float>* getSubCurDelta(std::string name){
		if(mapId.find(name) != mapId.end()){
			return curDelta[mapId[name]];
		}else{
			printf("BranchLayer: can not find getCurDelta %s\n", name.c_str());
			exit(0);
			return NULL;
		}
	}

	void setPreDelta(cuMatrix<float>* _preDelta){
		preDelta = _preDelta;
	}

	int getOutputAmount(){
		return outputAmount;
	}

	int getOutputDim(){
		return outputDim;
	}

	void printParameter(){};

private:
	cuMatrix<float>* inputs;
	cuMatrix<float>* preDelta;

	cuMatrixVector<float>outputs;
	cuMatrixVector<float>curDelta;
	
	std::map<std::string, int>mapId;

	int batch;
};
#endif
