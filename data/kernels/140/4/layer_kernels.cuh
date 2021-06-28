/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 October 13, 2013, modified by Li Sijin 
 */

#ifndef LAYER_KERNELS_CUH
#define	LAYER_KERNELS_CUH

#include <helper_cuda.h> //#include <cutil_inline.h>
#include <nvmatrix.cuh>

#define LOGREG_GRAD_THREADS_X      32
#define LOGREG_GRAD_THREADS_Y      4

#define LOGREG_ERR_THREADS_X        128
#define LOGREG_ERR_THREADS_Y        1

#define ELTLOGREG_ERR_THREADS_X        256
#define ELTL2SVM_ERR_THREADS_X 256
#define SSVM_ERR_THREADS_X 128

void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out);
void computeLogregGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);
void computeSoftmaxGrad(NVMatrix& acts, NVMatrix& actsGrad, NVMatrix& target, bool add);
void computeEltwiseLogregCost(NVMatrix& indmap, NVMatrix& predmap, NVMatrix & indlogpred, NVMatrix& correctprobs);
void computeEltwiseLogregGrad(NVMatrix& indmap, NVMatrix& predmap, NVMatrix& target, bool add, float coeff);
void computeEltwiseL2SVMCost(NVMatrix& labels, NVMatrix& y, NVMatrix & pre_grad, NVMatrix& all_cost, float a, float b);
void computeEltwiseL2SVMGrad(NVMatrix& labels, NVMatrix& pre_grad, NVMatrix& grad, bool add, float b, float coeff); 
// Numerical stability optimization: this routine combines computeLogregGrad with computeSoftmaxGrad
// to avoi dividing and then multiplying by quantities that may be near zero.
void computeLogregSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);
void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add);
void computeSSVMCost(NVMatrix& ind, NVMatrix& pred, NVMatrix& act_max_ind,\
                      NVMatrix& act_max_value);
#endif	/* LAYER_KERNELS_CUH */

