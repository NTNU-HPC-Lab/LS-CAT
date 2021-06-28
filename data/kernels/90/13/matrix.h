#ifndef _MATRIX_H
#define _MATRIX_H

extern "C"{
#include "Python.h"
#include "arrayobject.h"
}

class Matrix {
    private:
        int nrow, ncol;
        bool trans;
        bool own;
        float *data;
        void _init(int, int, float*);
    public:
        Matrix(PyArrayObject *);
        Matrix(int nrow, int ncol, float low, float upper);
        Matrix(int nrow, int ncol);
        Matrix(Matrix&);
        ~Matrix();
        inline float& operator()(int i, int j){
            return this->data[i * this->ncol + j];
        }
        int get_row_num();
        int get_col_num();
        int get_ele_num();
        void reshape(int, int);
        bool get_trans();
        float* get_data();
        bool equal_value(Matrix&);
        bool equal_value(Matrix&, float);
        bool check_nan();

        void assign(Matrix& target);
        void mat_init(float val);
        void mat_add(Matrix& m, float sb);
        void mat_add(Matrix& m, Matrix& target, float sa, float sb);
        void ele_scale(float);
        void ele_scale(float, Matrix&);
        void ele_add(float);
        void ele_add(float, Matrix&);
        void mat_sum(int axis, Matrix&);
        float ele_mean();
        void mat_mul(Matrix& m, Matrix& target);
        void mat_mul(Matrix& m);
};

#endif
