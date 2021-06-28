#ifndef _NVMATRIX_H
#define _NVMATRIX_H

#include "matrix.h"


class NVMatrix {
    private:
        int nrow, ncol;
        bool trans;
        bool own;
        float *data;
    public:
        NVMatrix(Matrix&);
        NVMatrix(int nrow, int ncol);
        ~NVMatrix();
        inline float& operator()(int i, int j){
            return this->data[i * this->ncol + j];
        }
        int get_row_num();
        int get_col_num();
        int get_ele_num();
        bool get_trans();
        float* get_data();
        void reshape(int, int);

        void copyFromHost(Matrix& source);
        void assign(Matrix&);
        void assign(NVMatrix&);
        void mat_init(float val);
        void mat_add(NVMatrix& m, float sb);
        void mat_add(NVMatrix& m, NVMatrix& target, float sa, float sb);
        void mat_mul(NVMatrix& m);
        void mat_mul(NVMatrix& m, NVMatrix& target);
        void ele_add(float val, NVMatrix& target);
        void ele_add(float val);
        void ele_scale(float scaler, NVMatrix& target);
        void ele_scale(float scaler);
        void mat_sum(int axis, NVMatrix &target);
        float ele_mean();
};

#endif
