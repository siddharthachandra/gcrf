#ifndef __CGD__
#define __CGD__
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include "cuda-helper/helper_functions.h"
#include "cuda-helper/helper_cuda.h"
int solveCGD_GPUarrays_reduced(float* d_val, int *d_row, int* d_col, int N, int nz, float* d_r, const double tol, const int max_iter, float* d_x,cublasHandle_t cublasHandle_,cusparseHandle_t cusparseHandle_){
    float a, b, na, r0, r1;
    float dot;
    float *d_p, *d_Ax;
    float alpha, beta, alpham1;
    int k;
    cublasHandle_t cublasHandle = cublasHandle_;
    cublasStatus_t cublasStatus;
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));
    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.0;
    cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

    cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
    k = 1;
    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
        cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaThreadSynchronize();
        k++;
    }
    int ret_val = 0;
    if (k > max_iter){
        fprintf(stderr,"@#$!$%\n");
        ret_val = 1;
    }
    cusparseDestroy(cusparseHandle);
    cudaFree(d_p);
    cudaFree(d_Ax);
    return ret_val;
}
int solveCGD_GPUarrays_reduced(double* d_val, int *d_row, int* d_col, int N, int nz, double* d_r, const double tol, const int max_iter, double* d_x,cublasHandle_t cublasHandle_,cusparseHandle_t cusparseHandle_){
    double a, b, na, r0, r1;
    double dot;
    double *d_p, *d_Ax;
    double alpha, beta, alpham1;
    int k;
    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = cublasHandle_;
    cublasStatus_t cublasStatus;
    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    checkCudaErrors(cusparseStatus);
    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);
    checkCudaErrors(cusparseStatus);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(double)));
    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.0;
    cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);
    cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
    k = 1;
    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasDscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus = cublasDaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
        cublasStatus = cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasDaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasDaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaThreadSynchronize();
        k++;
    }
    int ret_val = 0;
    if (k > max_iter){
        fprintf(stderr,"@#$!$%\n");
        ret_val = 1;
    }
    cusparseDestroy(cusparseHandle);
    cudaFree(d_p);
    cudaFree(d_Ax);
    return ret_val;
}
#endif
