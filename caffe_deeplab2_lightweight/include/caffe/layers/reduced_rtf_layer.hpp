#ifndef CAFFE_LOSS_LAYERS_HPP_
#define CAFFE_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>
#include <set>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
namespace caffe {

/**
 * @brief reduced RTF Layer 
 *
 * Author: Siddhartha Chandra
 * Based on Regression Tree Fields!
 *
 */
template <typename Dtype>
class reducedRTFLayer : public LossLayer<Dtype> {
 public:
  explicit reducedRTFLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void InitializeParameters();
  virtual void FillBlob(Blob<Dtype>& toFill,bool isRand,Dtype fillerConstant);
//  virtual void printBlob(Blob<Dtype*> toFill);

  virtual inline const char* type() const { return "reducedRTFLayer"; }
  //virtual inline const char* type() const { return "RTFLayer"; } //SID
  //virtual inline int ExactNumBottomBlobs() const {return 4;} //one blob for unary features, one blob for labels, one each for vertical, horizontal features.
  virtual inline int ExactNumBottomBlobs() const {return -1;} 
  virtual inline int MinBottomBlobs() const {return 3;} //one blob for unary features, one blob for labels, one each for vertical, horizontal features. More bottom blobs for more complicated connectivities.
  virtual inline int MinTopBlobs() const { return 1; }
  // RTF layer can only output one blob, which is the predicted label.
  virtual inline int MaxTopBlobs() const {
	return 1;}

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 3;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //These are the variables we need. Book-keeping.
  //Only need the number of channels, and number of labels! Unary coefficients are (#labels X #channels). Binary coefficients are (2 X (#labels X #channels X #labels) )
  /*
  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  */
  //The only parameter we need is the num_labels.
  int channels_;
  unsigned int num_labels,solver_type,dummy_labels,max_iter;
  int height_, width_;
  double etaU, weight_decay, momentum, learning_rate, etaBH, etaBV;
  int sample_num, update_at,num_train,num_test;
  double sum,tot,tot_loss,diagonal_element;
  double tot_loss_test, sum_test, tot_test;
  double tolerance;
  vector<float> tot_per_label;
  vector<float> sum_per_label;
  vector<int> offsets_x;
  vector<int> offsets_y;
  string weights_file;
  bool has_label;
  //Caffe blobs for internal algorithms.
  Blob<int> I_d;
  Blob<float> iscorrect_d;                      
  Blob<Dtype> rhs_d;                       
  Blob<float> losses_d;                         
  Blob<Dtype> x_d;                         
  Blob<Dtype> sum_x_d;                         
  Blob<Dtype> x_k;                         
  Blob<Dtype> sum_rhs_d;                         
  Blob<int> idxmap; 
  Blob<int> J_d;
  Blob<int> which_d;                                        
  Blob<int> where_d;                                        
  Blob<Dtype> val_d;  
  int unary_ind,label_ind,rtf_init_ind,numPW;
  //int pooled_height_, pooled_width_;
  //bool global_pooling_;
  //Don't know what the following are for. Will figure out later!
  //Blob<Dtype> rand_idx_;
  //Blob<int> max_idx_;
};


}  // namespace caffe

#endif  // CAFFE_LOSS_LAYERS_HPP_
