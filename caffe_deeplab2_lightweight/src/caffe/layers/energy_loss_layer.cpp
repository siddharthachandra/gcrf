#include <vector>
#include <cmath>
#include "caffe/layers/energy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EnergyLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  n_ = static_cast<int>(this->layer_param_.energy_loss_param().n()); 
}

template <typename Dtype>
void EnergyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //Nothing to be done. This is an energy based loss.
}

template <typename Dtype>
void EnergyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype * bottom_data = bottom[0]->mutable_cpu_data();
    Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
    for(int i = 0; i < bottom[0]->count(); ++i)
        bottom_diff[i] = (1/(pow(1 + bottom_data[i]*bottom_data[i],n_)));
}

#ifdef CPU_ONLY
STUB_GPU(EnergyLossLayer);
#endif

INSTANTIATE_CLASS(EnergyLossLayer);
REGISTER_LAYER_CLASS(EnergyLoss);

}  // namespace caffe
