#include <algorithm>
#include <vector>
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"  
#include "caffe/layers/reduced_rtf_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {


template <typename TypeParam>
class reducedRTFLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  reducedRTFLayerTest()
      : blob_bottom_1_(new Blob<Dtype>(1, 5, 3, 3)),
        blob_bottom_2_(new Blob<Dtype>(1, 1, 2, 3)),
        blob_bottom_3_(new Blob<Dtype>(1, 1, 3, 2)),
        blob_top_(new Blob<Dtype>()) {}

    virtual void SetUp() {
    Caffe::set_random_seed(1701);

    FillerParameter filler_param;
    filler_param.set_value(10.);

    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_1_);
    filler.Fill(this->blob_bottom_2_);
    filler.Fill(this->blob_bottom_3_);
    Dtype* blob_data_1 = blob_bottom_1_->mutable_cpu_data();
    Dtype* blob_data_2 = blob_bottom_2_->mutable_cpu_data();
    Dtype* blob_data_3 = blob_bottom_3_->mutable_cpu_data();

    // int d1 = blob_bottom_1_->count();
    // int d2 = blob_bottom_2_->count();

    // caffe_abs(d1, blob_data_1, blob_data_1);
    // caffe_abs(d2, blob_data_2, blob_data_2);

    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_3_);

    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~reducedRTFLayerTest() {
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_bottom_3_;
    delete blob_top_;
  }


  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_3_;
  Blob<Dtype>* const blob_top_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  };

TYPED_TEST_CASE(reducedRTFLayerTest, TestDtypesAndDevices);

TYPED_TEST(reducedRTFLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  
  LayerParameter layer_param;
  RTFLayerParameter* rtf_layer_param = layer_param.mutable_rtf_layer_param();
  rtf_layer_param->set_num_labels(5);
  rtf_layer_param->set_solver_type(0);
  rtf_layer_param->set_has_label(0);
  rtf_layer_param->add_offset_x(0);
  rtf_layer_param->add_offset_y(1);
  rtf_layer_param->add_offset_x(1);
  rtf_layer_param->add_offset_y(0);
  rtf_layer_param->set_eta_u(1);
  rtf_layer_param->set_eta_bh(1);
  rtf_layer_param->set_eta_bv(1);
  rtf_layer_param->set_diagonal_element(10);
  rtf_layer_param->set_num_train(1);
  rtf_layer_param->set_num_test(0);
  reducedRTFLayer<Dtype> layer(layer_param);

  // ConcatLayer<Dtype> layer(layer_param);
  // const Dtype* top_data = this->blob_top_->cpu_data();

   GradientChecker<Dtype> checker(1e-2, 1e-3);
   checker.CheckGradient( &layer, this->blob_bottom_vec_,  this->blob_top_vec_);
  
  // std::cout<<top_data[0];
  // layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  // const Dtype* top_data = this->blob_top_->cpu_data();

    // std::cout<<std::endl<<top_data[0]<<std::endl;

}

} 
