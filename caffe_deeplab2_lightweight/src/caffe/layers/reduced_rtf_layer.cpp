#include <algorithm>
#include <cfloat>
#include <ctime>
#include <vector>
#include <iostream>
#include <stdlib.h>
using namespace std;
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/reduced_rtf_layer.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
    void reducedRTFLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        dummy_labels = 1;
        max_iter = 10000;
        tolerance = 1e-5;
        RTFLayerParameter rtf_layer_param = this->layer_param_.rtf_layer_param();
        if(rtf_layer_param.has_weights_file()){
            weights_file=(string)rtf_layer_param.weights_file();
        }
        else
            weights_file="";
        if(rtf_layer_param.has_has_label()){
           has_label = (bool)rtf_layer_param.has_label(); 
        }
        else
            has_label = 0;
        CHECK_GT(rtf_layer_param.offset_x_size(),0) << "offset_x should have atleast one element!";
        CHECK_GT(rtf_layer_param.offset_y_size(),0) << "offset_y should have atleast one element!";
        offsets_x.resize(rtf_layer_param.offset_x_size(),0);
        offsets_y.resize(rtf_layer_param.offset_y_size(),0);
        for (int param_id = 0; param_id < offsets_x.size(); ++param_id)
            offsets_x[param_id] = rtf_layer_param.offset_x(param_id);
        for (int param_id = 0; param_id < offsets_y.size(); ++param_id)
            offsets_y[param_id] = rtf_layer_param.offset_y(param_id);
        num_train = (int)rtf_layer_param.num_train();
        num_test = (int)rtf_layer_param.num_test();
        num_labels = (int)rtf_layer_param.num_labels();
        solver_type = (int)rtf_layer_param.solver_type();
        etaU  =(double)rtf_layer_param.eta_u();
        diagonal_element =(double)rtf_layer_param.diagonal_element();
        etaBH  =(double)rtf_layer_param.eta_bh();
        etaBV  =(double)rtf_layer_param.eta_bv();
        sample_num = 0;
        sum = 0;
        tot = 0;
        tot_loss = 0;
        sum_test = 0;
        tot_test = 0;
        tot_loss_test = 0;
        InitializeParameters();
        sum_per_label.resize(num_labels,0);
        tot_per_label.resize(num_labels,0);
    }

//resize all arrays (blobs) for use
template <typename Dtype>
    void reducedRTFLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  //fprintf(stderr,"bottom[0] Shape %d %d %d %d\n",bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(),bottom[0]->width());
  //fprintf(stderr,"bottom[1] Shape %d %d %d %d\n",bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(),bottom[1]->width());
  //fprintf(stderr,"bottom[2] Shape %d %d %d %d\n",bottom[2]->num(), bottom[2]->channels(), bottom[2]->height(),bottom[2]->width());
  top[0]->Reshape(bottom[0]->num(), num_labels, height_, width_); //N X L X H X W (W = fastest dimension)
  //fprintf(stderr,"Finished Reshape\n");
}

template <typename Dtype>
void reducedRTFLayer<Dtype>::FillBlob(Blob<Dtype> &toFill, bool isRand, Dtype fillerConstant){

    int N,C,H,W;

    N = toFill.num();
    C = toFill.channels();
    H = toFill.height();
    W = toFill.width();
    Dtype* toFillPtr = toFill.mutable_cpu_data();
    for(int n=0; n<N; ++n){
        for(int c=0; c<C; ++c){
            for(int h=0; h<H; ++h){
                for(int w=0; w<W; ++w){
                    if(isRand){
                        toFillPtr[n*C*H*W + c*H*W + h*W + w] = (Dtype)(rand()/RAND_MAX);
                    }
                    else{
                        toFillPtr[n*C*H*W + c*H*W + h*W + w] = fillerConstant;
                    }
                }
            }
        }
    }

}

template <typename Dtype>
void reducedRTFLayer<Dtype>::InitializeParameters(){ //use FillBlob() to initialize any parameters you want.
}

template <typename Dtype>
void reducedRTFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    return;
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void reducedRTFLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
    NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(reducedRTFLayer);
#endif

INSTANTIATE_CLASS(reducedRTFLayer);
REGISTER_LAYER_CLASS(reducedRTF);
}  // namespace caffe
