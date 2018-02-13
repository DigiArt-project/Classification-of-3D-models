# 3D ResNets in Keras

Using Wide Residual Networks (WRNs) [1] and Residual of Residual Networks (RoRs) [2] for 3D object classification.

Experiments were based on the following implementations:
https://github.com/titu1994/Wide-Residual-Networks
https://github.com/titu1994/Residual-of-Residual-Networks

Data pre-processing was based on the code provided at:
https://github.com/dimatura/voxnet

Dataset used for training: 
- ModelNet10 (3dshapenets.cs.princeton.edu)

# Requirements
- Python (numpy, scipy, sklearn, tarfile, zlib)
- Theano
- Keras

# References
[1] https://arxiv.org/abs/1605.07146
[2] https://arxiv.org/abs/1608.02908
