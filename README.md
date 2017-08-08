# CCSC_code_ICCV2017
This is the source code repository for the ICCV 2017 paper "Consensus Convolutional Sparse Coding".

#### Authors: 

Biswarup Choudhury, Robin Swanson, Felix Heide, Gordon Wetzstein, and Wolfgang Heidrich.

#### Repository Information: 

All code was written and tested in MATLAB 2016b

1. 2D: Learning 2D convolutional filters from large image datasets, like ImageNet (to be downloaded separately). Also contains code for reconstruction problems such as inpainting and Poisson deconvolution using the filters learned. 

2. 2-3D: Learning convolutional filters for hyperspectral images. Also contains code for hyperspectral inpainting and demosaicing.

3. 3D: Learning 3D convolutional filters for video datasets (to be downloaded separately). Also contains code for video deblurring using the filters learned.

4. 4D: Learning 4D filters for lightfield datasets (sample input lightfield data provided). Also contains code for novel view synthesis using filters learned.

5. image_helpers: Miscellaneous utility code for reading data, contrast normalization, etc.

#### Memory Requirements:

All experiments were conducted under 128GB of memory.

#### Reference:

If you use any of the above code or a version inspired by it, please cite our paper. Thank you!

```
@Article{Choudhury:2017:CCSC,
  author =       {B. Choudhury and R. Swanson and F. heide and G. Wetzstein and W. Heidrich},
  title =        {Consensus Convolutional Sparse Coding},
  journal =      {IEEE Xplore (Proc. ICCV)},
  year =         2017,
}
```

For bugs, questions and comments, please send email to:

1. Biswarup Choudhury [biswarup.c@gmail.com]
   Research Scientist, VCC, KAUST

2. Robin Swanson [robin@cs.toronto.edu]
   PhD Student, University of Toronto

All the best :-)

