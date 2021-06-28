//
// Created by LZR on 2019/1/4.
//
#pragma once
#define ERRORCODE -99999999
#define INTEGER 0
#define FLOAT 1
#define DOUBLE 2
#define LONGLONG 3


#define NORMALMATRIXMUL 873
#define NORMALMATRIXMULPARALLEL 389
#define ALGOSTRASSEN 648
#define ALGOSTRASSENPARALLEL 933
#define ALGODNS 837
#define ALGOCANNON 926
#define NOALGO 742


#include <vector>
class Matrix {
public:
	Matrix();
	Matrix(int Type);
	Matrix(int row, int col, int type);
	Matrix *generateMatrixParts(int leftTopX, int leftTopY, int rightDownX, int rightDownY);
	void readMatrix();
	void writeMatrix(std::string fileName);
	void randomMatrix(int row, int col, int matrixType, int MIN, int MAX);
	void printMatrix();
	void initVectorSpace();
	Matrix* normalMatrixMultiple(Matrix* outMatrix);
	int getRow();
	int getCol();
	void setRow(int row);
	void setCol(int col);
	void setType(int type);
	int getType();
	void* returnVectorData();
	void changeType(int matrixType);
	void clearTypeMatrix();
	double getMatrixElement(int x, int y);
	void setMatrixElement(int x, int y, double val);
	void addIntoMatrixElement(int x, int y, double val);
	double getMatrixElement(int i);
	void setMatrixElement(int i, double val);
	void matrixAdd(Matrix* outMatrix);
	void matrixSub(Matrix* outMatrix);
	void matrixPush(double x);
	bool matrixCompare(Matrix *outMatrix);
	~Matrix();


private:
	int row, col;
	int matrixType;

	std::vector<int> integerMatrix;
	std::vector<float> floatMatrix;
	std::vector<double> doubleMatrix;
	std::vector<long long> longMatrix;
};


