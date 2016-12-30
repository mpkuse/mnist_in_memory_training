# MNIST - In Memory with Caffe Example

From the MNIST data set (as image files), read say 128 images at a time (batch size) using opencv
and feed them into caffe (pycaffe)

The basic idea of this example is that, it shows how to feed data into caffe from an array. 

In this repository, I am not including the MNIST dataset, please download from the official site.
Make a folder in this directory name it `train/` and save all files as .png files. Also
make a annotation.txt file. Each line of this file contains a file-path and number (label).
./train/0.png 5<br/>
./train/1.png 0<br/>
./train/2.png 4<br/>
./train/3.png 1<br/>
./train/4.png 9<br/>
./train/5.png 2<br/>
./train/6.png 1<br/>
./train/7.png 3<br/>
./train/8.png 1<br/>
./train/9.png 4<br/>

Finally, I recommend reading `caffe_train_in_memory.py` to understand how to do it. 

## Author
Manohar Kuse <mpkuse@ust.hk> <br/>
July-Aug, 2016
