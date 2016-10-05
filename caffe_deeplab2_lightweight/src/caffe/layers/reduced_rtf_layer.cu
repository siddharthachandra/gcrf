/***************************************************
  * Author: Siddhartha Chandra
  * TODO: CuSparse Handle to be invoked once from
  * the caffe() constructor. 
  *************************************************/

#include <algorithm>
#include <cfloat>
#include <ctime>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
using namespace std;
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/reduced_rtf_layer.hpp"
#include "CGDsolve/conjugateGradient_reduced.cpp"

//#define DEBUG
namespace caffe {
template <typename Dtype>
__global__ void getA_part_reduced(int nz, const Dtype* binary, int absIndex, Dtype *val, int*which, int*where,double diagonal_term){
    CUDA_KERNEL_LOOP(index,nz){
        if(which[index] == -1 && absIndex == -1){ //diagonal
            val[index] = -1.0f;
        }
        else if(absIndex == which[index]){ //horizontal
            val[index] = -(Dtype)binary[where[index]]/(Dtype)(diagonal_term);
        }
    }
}

template <typename Dtype>
__global__ void setBinaryDiff_reduced(int numElements, int W, int height, int width, int thisnumvariables, int num_labels, int offset, Dtype* binary_diff, Dtype* dldb, Dtype* x,double ETA){
    int w,var,h,c,c1,c2;
    CUDA_KERNEL_LOOP(index,numElements){
        w = index%width;
        var = index%thisnumvariables;
        h = (var-w)/width;
        c = (index-var)/thisnumvariables;
        c1 = c/num_labels;
        c2 = c%num_labels;
        var = h*W + w;
        binary_diff[index] = binary_diff[index]+1*ETA*dldb[var*num_labels+c1]*x[(var+offset)*num_labels+c2]+1*ETA*dldb[(var+offset)*num_labels+c2]*x[var*num_labels+c1];
    }
}

template <typename Dtype>
void reducedRTFLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    /* We assume symmetric offsets in both X and Y directions.
       for instance, if offset_x=1, we will have pairwise interactions of a cell with the immediate left and immediate right cells.
    */
    numPW = offsets_x.size();
    CHECK_EQ(numPW, offsets_y.size()) << "Offsets x,y not equal in number."<<endl;

    /* Bottom Organization => Unary:: Pairwise */ 
 
    unary_ind = 0;
    rtf_init_ind = 0;
    label_ind = -1;
    const Dtype* unary = bottom[unary_ind]->gpu_data();
    const Dtype* binary_horz_gpu = bottom[1]->gpu_data();
    const Dtype* binary_vert_gpu = bottom[2]->gpu_data();
    const Dtype* labels_gpu;
    const Dtype* labels;
    int haveLabels = has_label;
    int count = bottom[unary_ind]->count();
    int channel = bottom[unary_ind]->channels();
    Dtype* unary_diff_gpu = bottom[unary_ind]->mutable_gpu_diff();
    Dtype* binary_horz_diff_gpu = bottom[1]->mutable_gpu_diff();
    Dtype* binary_vert_diff_gpu = bottom[2]->mutable_gpu_diff();
    Dtype* top_data_gpu = top[0]->mutable_gpu_data();

    int numSamples = bottom[unary_ind]->num();
    int W = bottom[unary_ind]->width();
    int H = bottom[unary_ind]->height();
    int numVariables = H*W,thisI;
    int n,c,h,w,c1,c2,var;
    int N,row,col,elem,solver_success;
    int sizeSolutionSpace = dummy_labels*numVariables;
    int num_labels2 = dummy_labels;

    CHECK_EQ(num_labels, bottom[unary_ind]->channels()) << "Unary Feature Dimensions are not Equal to #Labels"<<endl;
    for(int ii=0; ii<numPW; ++ii){
  //      cout<<"Offset: ("<<offsets_x[ii]<<","<<offsets_y[ii]<<") :: "<<bottom[ii+1]->num()<<" x "<<bottom[ii+1]->channels()<<" x "<<bottom[ii+1]->height()<<" x "<<bottom[ii+1]->width()<<endl;
        CHECK_EQ(bottom[ii+1]->channels(),num_labels2)<<"Size of offset: ("<<offsets_x[ii]<<","<<offsets_y[ii]<<") Pairwise Features wrong"<<endl;
        CHECK_EQ(W-abs(offsets_x[ii]),bottom[ii+1]->width())<<"Width of offset: ("<<offsets_x[ii]<<","<<offsets_y[ii]<<") Pairwise Features wrong"<<endl;
        CHECK_EQ(H-abs(offsets_y[ii]),bottom[ii+1]->height())<<"Height of offset: ("<<offsets_x[ii]<<","<<offsets_y[ii]<<") Pairwise Features wrong"<<endl;
    }

    sample_num += numSamples;
    N = numSamples; int C = num_labels2;

    /* ------------------ STEP 0: Prerequisites ----------------------*/
    //Resizing some blobs.
    rhs_d.Reshape(1,1,1,sizeSolutionSpace); // = numVariables
    x_k.Reshape(1,1,1,sizeSolutionSpace); // = numVariables
    sum_rhs_d.Reshape(1,1,1,numVariables);
    x_d.Reshape(1,1,1,sizeSolutionSpace); // = numVariables
    sum_x_d.Reshape(1,1,1,sizeSolutionSpace); // = numVariables

    /* ------------------ STEP 0.1 : Fill I -------------------------*/
    I_d.Reshape(1,1,1,sizeSolutionSpace+1);
    elem = 0; I_d.mutable_cpu_data()[elem]=0;
    for(h=0; h<H; ++h){
        for(w=0; w<W; ++w){
            thisI = 2*(numPW)*dummy_labels + 1;
            /***
             We are counting the number of non-zero entries per row.
             Each pairwise term gives rise to 2 neighbours (symmetric), for instance offset_x=1 gives left and right neighbours.
             Each neighbour gives rise to num_labels terms. The diagonal is always present. 
             Therefore (2*numPW*num_labels+1) is the maximum number of (upper bound on) terms that can be present in a row.
             Now, whenever a neighbour is absent, we reduce *num_labels* terms from this number.
             ***/
            for(int param_id=0; param_id<numPW; ++param_id){
                //neighbour 1 (w-o_x,h-o_y)
                if(h-offsets_y[param_id]<0 || w-offsets_x[param_id]<0 || h-offsets_y[param_id]>=H || w-offsets_x[param_id]>=W) thisI -= dummy_labels;
                //neighbour 2 (w+o_x,h+o_y)
                if(h+offsets_y[param_id]<0 || w+offsets_x[param_id]<0 || h+offsets_y[param_id]>=H || w+offsets_x[param_id]>=W) thisI -= dummy_labels; 
            }
            for(c=0; c<dummy_labels; ++c){
                ++elem;
                I_d.mutable_cpu_data()[elem] = I_d.mutable_cpu_data()[elem-1] + thisI;
            }
        }
    }
    CHECK_EQ(elem,sizeSolutionSpace);
    /* ------------------ STEP 0.2 : CSR FORMAT for A --------------*/
    int nz = I_d.cpu_data()[sizeSolutionSpace]; 
    J_d.Reshape(1,1,1,nz);
    which_d.Reshape(1,1,1,nz);
    where_d.Reshape(1,1,1,nz);
    val_d.Reshape(1,1,1,nz);
    /* ------------------ STEP 0.2.1 : which&where ----------------*/
    row = -1; elem=-1;
    for(h=0; h<H; ++h){
        for(w=0; w<W; ++w){
            for(c1=0; c1<dummy_labels; ++c1){
                ++row;
                //offsets (in cyclic order): (1,1 :: \ diagonal) (0,1 :: bottom,top) (-1,1 :: / diagonal) (1,0 :: right,left)
                for(int param_id=0; param_id<numPW; ++param_id){
                    if(h-offsets_y[param_id]>=0 && w-offsets_x[param_id]>=0 && h-offsets_y[param_id]<H && w-offsets_x[param_id]<W){
                        var = (h-offsets_y[param_id])*W + w-offsets_x[param_id]; //neighbour, always < (h,w)
                        col = var*dummy_labels;
                        for(c2=0; c2<dummy_labels; ++c2){
                            ++elem;
                            which_d.mutable_cpu_data()[elem] = param_id;
                            /*** for the -offset neighbours, the pairwise terms come from (h-offset,w-offset)
                              Therefore, the pairwise terms are symmetric.
                             PW(h+o_y,w+o_x,h,w) ==> (h,w)
                             ***/
                            where_d.mutable_cpu_data()[elem] = (c2*dummy_labels+c1)*(H-offsets_y[param_id])*(W-abs(offsets_x[param_id])) + (h-offsets_y[param_id])*(W-abs(offsets_x[param_id])) + w-offsets_x[param_id];
                            J_d.mutable_cpu_data()[elem] = col; 
                            ++col;
                        }
                    }
                }
                //place Identity Element
                ++elem;
                which_d.mutable_cpu_data()[elem] = -1;
                where_d.mutable_cpu_data()[elem] = -1;
                J_d.mutable_cpu_data()[elem] = row;
                for(int param_id=numPW-1; param_id>=0; --param_id){
                    if(h+offsets_y[param_id]>=0 && w+offsets_x[param_id]>=0 && h+offsets_y[param_id]<H && w+offsets_x[param_id]<W){
                        var = (h+offsets_y[param_id])*W + w+offsets_x[param_id]; //neighbour, always > (h,w)
                        col = var*dummy_labels;
                        for(c2=0; c2<dummy_labels; ++c2){
                            ++elem;
                            which_d.mutable_cpu_data()[elem] = param_id;
                            /*** for the +offset neighbours, the pairwise terms come from (h,w)
                              Therefore, the pairwise terms are symmetric.
                             PW(h,w,h+o_y,w+o_x) ==> (h,w)
                             ***/
                            where_d.mutable_cpu_data()[elem] = (c1*dummy_labels+c2)*(H-offsets_y[param_id])*(W-abs(offsets_x[param_id])) + (h)*(W-abs(offsets_x[param_id])) + w;
                            J_d.mutable_cpu_data()[elem] = col; 
                            ++col;
                        }
                    }
                }
            } //c1<dummy_labels
        }//w<W
    }//h<H //gotten A!!!
    /* ------------------ STEP 0.3 : GET CUBLAS,CUSPARSE HANDLES ---*/
    cublasHandle_t cublasHandle = Caffe::cublas_handle();
    cusparseHandle_t cusparseHandle;
    n=0;
    //setting initialization to unaries..
    caffe_copy(top[0]->count(),bottom[unary_ind]->gpu_data(),top[0]->mutable_gpu_data());
    CHECK_EQ(1,N)<<"batch size should be 1, otherwise we will have to recompute things...";

    for(n=0; n<N; ++n) { //iterating over the samples.
        /* ------------------- STEP 1: getting A ------------------------- */
        //getA diagonal
        getA_part_reduced<<<CAFFE_GET_BLOCKS(nz),CAFFE_CUDA_NUM_THREADS>>>(nz,bottom[0]->gpu_data(),-1,val_d.mutable_gpu_data(),which_d.mutable_gpu_data(),where_d.mutable_gpu_data(),diagonal_element);
        CUDA_POST_KERNEL_CHECK;
        //getA Pairwise 
        for(int ii=0; ii<numPW; ++ii){
            getA_part_reduced<<<CAFFE_GET_BLOCKS(nz), CAFFE_CUDA_NUM_THREADS>>>(nz,bottom[ii+1]->gpu_data()+bottom[ii+1]->offset(n),ii,val_d.mutable_gpu_data(),which_d.mutable_gpu_data(),where_d.mutable_gpu_data(),diagonal_element); 
        CUDA_POST_KERNEL_CHECK;
        }
        /* ------------------- STEP 2: getting B ------------------------- */
        //get B = (1 / (L - 1) ) \sum_i B_i
        caffe_gpu_set(numVariables,(Dtype)0.0,sum_rhs_d.mutable_gpu_data());
        for (int label=0; label<num_labels; ++label){
            caffe_gpu_axpy(numVariables,(Dtype)(1.0),bottom[unary_ind]->gpu_data()+bottom[unary_ind]->offset(n,label),sum_rhs_d.mutable_gpu_data());
        }
    sum_rhs_d.scale_data(Dtype(1.0/(num_labels-1)));
        /* ----------------- STEP 3: solving for X ----------------------- */
        for(int label=0; label<num_labels; ++label){
            //get rhs rhs_d = B - B_k
            caffe_copy(numVariables,bottom[unary_ind]->gpu_data()+bottom[unary_ind]->offset(n,label),rhs_d.mutable_gpu_data());
            rhs_d.scale_data(-1.0);
            caffe_gpu_axpy(numVariables,(Dtype)(1.0),sum_rhs_d.gpu_data(),rhs_d.mutable_gpu_data());
            //solve A x_k = rhs_d
            solver_success = solveCGD_GPUarrays_reduced(val_d.mutable_gpu_data(),I_d.mutable_gpu_data(),J_d.mutable_gpu_data(),sizeSolutionSpace,nz,rhs_d.mutable_gpu_data(),tolerance,max_iter,top[0]->mutable_gpu_data()+top[0]->offset(n,label),cublasHandle,cusparseHandle);
            if(solver_success){ //solver failed. Set solution to unaries. 
		        caffe_copy(numVariables,bottom[unary_ind]->gpu_data()+bottom[unary_ind]->offset(n,label),top[0]->mutable_gpu_data()+top[0]->offset(n,label));
            }
        }
    } //n<N
}

template <typename Dtype>
void reducedRTFLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        const Dtype* unary = bottom[unary_ind]->gpu_data();
        int count = bottom[unary_ind]->count();
        int channel=bottom[unary_ind]->channels();
        Dtype* unary_diff_gpu = bottom[unary_ind]->mutable_gpu_diff();
        const Dtype* top_diff_gpu = top[0]->gpu_diff();
        int numSamples = bottom[unary_ind]->num();
        int W = bottom[unary_ind]->width();
        int H = bottom[unary_ind]->height();
        int numVariables = H*W;
        int n,thisnumvariables,numelements;
        int N,solver_success;
        int sizeSolutionSpace = dummy_labels*numVariables;
        int num_labels2 = dummy_labels;
        N = numSamples;
        int nz = I_d.cpu_data()[sizeSolutionSpace]; 

        /* ------------------ STEP 0.3 : GET CUBLAS,CUSPARSE HANDLES ---*/
        cublasHandle_t cublasHandle = Caffe::cublas_handle();
        cusparseHandle_t cusparseHandle; // = Caffe::cusparse_handle();
        n=0;
        for(n=0; n<N; ++n) { //iterating over the samples.
            //get X = \sum_i x_i
            caffe_gpu_set(numVariables,(Dtype)0.0,sum_x_d.mutable_gpu_data());
            for (int label=0; label<num_labels; ++label){
                caffe_gpu_axpy(numVariables,(Dtype)(1.0),top[0]->gpu_data()+top[0]->offset(n,label),sum_x_d.mutable_gpu_data());
            }
            //get diff(X) = \sum_i diff(x_i)
            caffe_gpu_set(numVariables,(Dtype)0.0,sum_x_d.mutable_gpu_diff());
            for (int label=0; label<num_labels; ++label){
                caffe_gpu_axpy(numVariables,(Dtype)(1.0),top[0]->gpu_diff()+top[0]->offset(n,label),sum_x_d.mutable_gpu_diff());
            }
		    /* ----------------- STEP 4.2: computing dL_dB ------------------- */
            //get unary diffs now
            // A (dL/dB_k) = dL/d(x_k) where x_k = \sum_{i~=k} x_i
    		for(int label=0; label<num_labels; ++label){

		            //get rhs dldx_k = X - x_k
		            caffe_copy(numVariables,top[0]->gpu_diff()+top[0]->offset(n,label),x_k.mutable_gpu_diff());
                    x_k.scale_diff(1.0-num_labels);
		            caffe_gpu_axpy(numVariables,(Dtype)(1.0),sum_x_d.gpu_diff(),x_k.mutable_gpu_diff());
		            x_k.scale_diff(Dtype(1.0/(num_labels-1)));
		            //solve (A dL/dB_k) = dL/dx_k where  x_k = \sum_{i~=k} x_i
		            solver_success = solveCGD_GPUarrays_reduced(val_d.mutable_gpu_data(),I_d.mutable_gpu_data(),J_d.mutable_gpu_data(),sizeSolutionSpace,nz,x_k.mutable_gpu_diff(),tolerance,max_iter,bottom[unary_ind]->mutable_gpu_diff()+bottom[unary_ind]->offset(n,label),cublasHandle,cusparseHandle);
                    if(solver_success){ //solver failed. Set all gradients to zero and exit function.
                        for(int ii=0; ii<bottom.size(); ++ii)
                            caffe_gpu_set(bottom[ii]->count(),Dtype(0.0),bottom[ii]->mutable_gpu_diff());
                        return;
                    }
		        caffe_copy(numVariables,top[0]->gpu_data()+top[0]->offset(n,label),x_k.mutable_gpu_data());
		        x_k.scale_data(-1.0);
		        caffe_gpu_axpy(numVariables,(Dtype)(1.0),sum_x_d.gpu_data(),x_k.mutable_gpu_data());
                // get dl_dA now.
                for(int ii=0; ii<numPW; ++ii){
                    thisnumvariables = (H-offsets_y[ii])*(W-abs(offsets_x[ii]));
                    numelements = thisnumvariables*dummy_labels;
                    setBinaryDiff_reduced<<<CAFFE_GET_BLOCKS(numelements),CAFFE_CUDA_NUM_THREADS>>>(numelements,W,H-offsets_y[ii],W-offsets_x[ii],thisnumvariables,dummy_labels,W*offsets_y[ii]+offsets_x[ii],bottom[ii+1]->mutable_gpu_diff()+bottom[ii+1]->offset(n),bottom[unary_ind]->mutable_gpu_diff()+bottom[unary_ind]->offset(n,label),x_k.mutable_gpu_data(),etaBV/diagonal_element);
                }
		     }
            /* ----------------- STEP 5.2: Set Binary Derivatives ----- */
        } //n<N
}

INSTANTIATE_LAYER_GPU_FUNCS(reducedRTFLayer);

}  // namespace caffe
