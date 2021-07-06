### Code for the TCI 2020 Paper
Multi-Angular Epipolar Geometry Based Light Field Angular Reconstruction Network

This code borrows heavily from the paper:

Fast Light Field Reconstruction With Deep Coarse-To-Fine Modeling of Spatial-Angular Clues

### Description

This code is used to generate an 7 ¡Á 7 densely-sampled LF with 3 ¡Á 3 LF views as inputs

### Requirements and Dependencies

- MATLAB
- cuda and cudnn (For GPU. Please modify install.m if not using cudnn)
- matconvnet (Please use the matconvnet code given in this repository. It contains the 3D convolution code written by authors)


### Training

Set the training and validation data directory (opts.test_dir) in init_opts.m. Download the training and validation datasets to the specofoc directories. Make sure that there are enough memory for loading the whole training and validatoin datasets.

    >> train

### Testing Pretrained Models

Set the testing data directory (opts.test_dir) in init_opts.m

    >> test

### Testing Your Own Models

    >> test_model(name, depth, gpu, saveImg, epoch, len)
    
- model_name    : model name
- depth         : model depth
- gpu           : GPU ID
- saveImg       : Save the HR SAIs if true
- epoch         : model epoch to test
- len           : controls the size of the sub-lightfield, value depends on GPU memory
   
### Authors of the Paper

 Deyang Liu , Yan Huang , Qiang Wu , Ran Ma, and Ping An 