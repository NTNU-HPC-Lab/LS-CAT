//
// Created by LZR on 2019/2/2.
//
#pragma once

#include "Matrix.h"

class MatrixCalculation {
public:
	static Matrix* matrixAdd(Matrix *leftMatrix, Matrix *rightMatrix);
	static Matrix* matrixSub(Matrix *leftMatrix, Matrix *rightMatrix);
	static Matrix* matrixMul(Matrix *leftMatrix, Matrix *rightMatrix);
	static Matrix* matrixMulParallel(Matrix *leftMatrix, Matrix *rightMatrix, int coreNum);

	static Matrix* algorithmStrassen(Matrix *matrixA, Matrix *matrixB, int coreNum, int mulCode);
	static Matrix* Strassen(Matrix *matrixA, Matrix *matrixB);

	static Matrix* StrassenParallel(Matrix *matrixA, Matrix *matrixB, int coreNum, int mulCode);

	static Matrix* algorithmCannon(Matrix *matrixA, Matrix *matrixB, int coreNum, int mulCode);
	static Matrix* Cannon(Matrix *matrixA, Matrix *matrixB, int matrixSideDivision, int mulCode);

	static Matrix* algorithmDNS(Matrix *matrixA, Matrix *matrixB, int coreNum, int mulCode);
	static Matrix* DNS(Matrix *matrixA, Matrix *matrixB, int threadCubeDivision, int mulCode);
	static Matrix* matrixMulAndInsertByBlock(Matrix *matrixA, Matrix *matrixB, Matrix *matrixC, int blockIdA, int blockIdB, int sideDivision, int coreNum);
	static int matrixTypeDecision(int typeA, int typeB);
	static Matrix* expandMatrixWithZero(Matrix *outMatrix, int newRow, int newCol);

	using TYPENOW = int;
	static int TYPEINT;
	static float TYPEFLOAT;
	static double TYPEDOUBLE;
	static long long TYPELONG;
};
