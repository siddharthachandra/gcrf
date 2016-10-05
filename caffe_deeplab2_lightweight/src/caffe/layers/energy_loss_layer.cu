#include <vector>
#include <cmath>
#include "caffe/layers/energy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
__global__ void derivative(int num, const Dtype* bottom, Dtype* diff_, int n_){
    CUDA_KERNEL_LOOP(index,num){
        diff_[index] = (1/(pow(1 + bottom[index]*bottom[index],n_)));
    }
}
template <typename Dtype>
void EnergyLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //Nothing to be done. This is an energy based loss.
}

template <typename Dtype>
void EnergyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   Dtype * bottom_data = bottom[0]->mutable_gpu_data();
   Dtype * bottom_diff = bottom[0]->mutable_gpu_diff();
   derivative<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(),bottom_data,bottom_diff,n_);
}

INSTANTIATE_LAYER_GPU_FUNCS(EnergyLossLayer);

}  // namespace caffe
