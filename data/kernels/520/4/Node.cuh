#include <map>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime> 

#pragma once
class Node
{
private:
	void setStartEndVertexes();
	int Node::getArrayLength(int *row);
	int Node::getArrayMinValue(int restrictVal, int row);
	//std::vector<int> rows;
	//std::vector<int> cols;
	//std::vector<int> localCycle;
	int *translateX;
	int *translateY;

	int *optimalX;
	int *optimalY;

public:
	Node(int size, int s, int s0);

	int baseSize;
	int points;
	int size;
	int S0;
	int S;
	int *M;
	int *P;
	
	void testMatrixAdduction(int *M);

	void copySessionDescription(int *rowD, int *colD);
	int getRealElement(int *dataDescription, int ind);


	void setP(int size, int *source);
	void setInitials(int size);
	void printMatrix();
	void subMinRowsAndCorrect();
	void subMinColsAndCorrect();



	
	void getPathForRemove(int &i, int &j);
	void setInfinityFor(int row, int col);
	void setMatrix(int *M);
	//void setMatrix(matrix<int> M);
	void setMatrixWithExclude(int *source, int excludeRow, int excludeCol);
	void setPreviouseNode(Node *prevNode);
	void invokeAdduction();
	void copyCandidatesP(int *P);

	void setInitialMatrix(int *sourceMatrix);


	void setMatrixWithRemoveExclude( int *source, int excludeRow, int excludeCol);
	void copyRowColDescriptors(std::vector<int> rows, std::vector<int> cols);

	int getHead(int tail);
	int getTail(int head);
	void handlePodcycles(int &a, int &b);
	void handleStraightforwardMatrix();

	Node* leftBranching(int row, int col);
	Node* rightBranching(int row, int col);
	bool isRowRemove(int *row);

	static const int InfityMaxValue = 1000000;
	static const int InfityMinValue = -1000000;

	int getRealSize();
	void setRealSize();
	void printArray(int size, int *arr);


	int* subMinRowsAndCorrect(int s, const int *row, const int size);

	int * cudaCleanCopy(int *source);

	~Node();
};

