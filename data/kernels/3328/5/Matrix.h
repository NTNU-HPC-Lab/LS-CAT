#pragma once

#include <iostream>

class Matrix
{
public:
	int _width;
	int _height;
	float* _elements;

	//Constructors
	Matrix(int width, int height);
	Matrix(const Matrix &matrix);

	//Deconstructor
	~Matrix();

	//Initialize Matrix
	//If randomize flag is set to true, matrix is filled with random values, else it is filled with num value.
	void Initialize(float num = 0, bool randomize = true);
	//Print Matrix
	void Print();


	//Operators
	Matrix operator+(const Matrix &matrix);
	Matrix operator-(const Matrix &matrix);
	Matrix operator*(const Matrix &matrix);
	const Matrix& operator=(const Matrix &matrix);
	float& operator[](int i);
	const float& operator[](int i) const;

	//Return transposition of matrix
	Matrix Transpose();
	//Return sum of all matrix elements
	float VectorSum();
	//Compare two matrices
	static void MatrixCompare(Matrix &A, Matrix &B);

private:

};
