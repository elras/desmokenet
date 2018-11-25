
## Deep Smoke Removal from Minimally Invasive Surgery Videos

We employed transfer learning, and reached state of the art performance and speed in automatic smoke removal from image-guided surgery videos (right-half enhanced by our method):

<div align="center">
  <img src="/videos/example1.gif" width="400" height="300"><br>
</div>


### Quick Demo

For a dirty demo, first install Anaconda then create a desmokenet environment
    
    $ conda create -n desmokenet

Install cpu-only pycaffe by (better not to use other channels: https://github.com/conda-forge/caffe-feedstock/issues/31)

    $ conda install -c defaults caffe -n desmokenet

Simply run fallowing inside /test_code

    $ python test.py
    
From the input images at data/img, output images will appear at data/result. The network is also tested with compiled Caffe framework in Ubuntu 16.04 system with CUDA 8.0. Note that, this code is just for demo purposes and may not show the performance stated in the paper. 

### Citing

S. Bolkar, C. Wang, F. A. Cheikh and S. Yildirim, "Deep Smoke Removal from Minimally Invasive Surgery Videos," _2018 25th IEEE International Conference on Image Processing (ICIP)_, Athens, Greece, 2018, pp. 3403 3407.   doi: 10.1109/ICIP.2018.8451815  

	@INPROCEEDINGS{bolkar2018,  
		author={S. Bolkar and C. Wang and F. A. Cheikh and S. Yildirim},  
		booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},  
		title={Deep Smoke Removal from Minimally Invasive Surgery Videos},  
		year={2018},  
		pages={3403-3407},  
		doi={10.1109/ICIP.2018.8451815},  
		ISSN={2381-8549},  
		month={Oct},}

### References

Our fast implementation is based on AOD-Net, please check their work at: https://arxiv.org/abs/1707.06543 

