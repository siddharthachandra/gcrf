# gcrf
Fast, Exact and Multi-Scale Inference for Semantic Image Segmentation with Deep Gaussian CRFs
Find the pdf of the paper [here](https://siddharthachandra.github.io/resources/chandra-eccv-2016.pdf).

    @article{ChandraEccv2015,
    title={Fast, Exact and Multi-Scale Inference for Semantic Image Segmentation with Deep Gaussian CRFs},
    author={Siddhartha Chandra and Iasonas Kokkinos},
    journal={ECCV},
    year={2015}
    }

Follow these steps to reproduce our 80.2 IoU on VOC 2012 test set.

1. Change the PATH to the VOC Dataset in scripts/resources/test_rtf_release.prototxt, scripts/flip_images.m, scripts/apply_dense_crf.m, scripts/average_lr.m
2. Compile caffe (caffe_deeplab2_lightweight)
3. Compile dense-crf (scripts/resources/densecrf)
4. For the remainder of the steps, go into the scripts directory as all paths in the scripts are relative.
    cd scripts
5. Download the trained caffemodel from [here](http://cvn.ecp.fr/personnel/siddhartha/resources/finetuned_iter_10000.caffemodel), and place it into the scripts/resources directory.
    * wget http://cvn.ecp.fr/personnel/siddhartha/resources/finetuned_iter_10000.caffemodel
	* mv finetuned_iter_10000.caffemodel resources
6. Flip test images
	* We flip test images horizontally, and then average the scores. Use the matlab script: scripts/flip_images.m
7. Score images using trained model.
	* Use the bash script scripts/score_images.sh
	* The results are written to scripts/results/release
8. Average the flipped scores, generate results without crf.
	* Use the matlab script: scripts/average_lr.m
	* The averaged scores are written to scripts/results/scores
	* The segmentation maps without dense CRF are written to scripts/results/nocrf
	* The segmentation maps without dense CRF achieve 79.5 mean pixel IoU on VOC2012 test set.
	* The results are [here](http://host.robots.ox.ac.uk:8080/anonymous/BWYMCO.html)
9. Apply dense CRF for object edge refinement.
	* Use the matlab script: scripts/apply_dense_crf.m
	* The densecrf post processed segmentations are written to scripts/results/crf
	* The densecrf post processed segmentation maps achieve 80.2 mean pixel IoU on VOC2012 test set.
	* The results are [here](http://host.robots.ox.ac.uk:8080/anonymous/UWGAFB.html)
