# Introduction
This is a cplusplus translation in caffe of google's deepdream python code. Any suggestion is welcome.

## TODO
1. This is currently way slower than numpy code. I tried using caffe blas api. Still now it is not working. 
2. The prototxt file can't have innerproduct layer. I deleted the last two layers of googlenet to make it work. See http://stackoverflow.com/questions/42634179/caffenet-reshape
