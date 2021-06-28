//
// Created by root on 23/03/2020.
//

#ifndef HELLOCUDA_MATRIX_CUH
#define HELLOCUDA_MATRIX_CUH


class Matrix {

public:

    int Rows = 0;
    int Columns = 0;
    double *Value = {};
    Matrix(int cols, int rows, double v[]);

    void print();
    Matrix multiply(Matrix m);
    Matrix multiplyScalar(double m);
    Matrix exp();
    void randomFill(double seed);
    void randomFillSmall();
    double sumAll();
    Matrix divideScalar(double m);
    Matrix softmax();
    Matrix clip(double min, double max);
    void zeros();
    Matrix transpose();
    Matrix add(Matrix m);
    Matrix sub(Matrix m);
    Matrix hadamard(Matrix m);
    Matrix subScalar(double m);
    Matrix subScalarInverse(double m);
    Matrix addScalar(double m);
    Matrix sigmoid();
    Matrix logit();
    Matrix tanh();


};



#endif //HELLOCUDA_MATRIX_CUH
