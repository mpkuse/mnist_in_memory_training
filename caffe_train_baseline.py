import caffe
import cv2
import time
import numpy as np


caffe.set_mode_gpu()
caffe.set_device(0)

solver = None
solver = caffe.SGDSolver('solver_baseline.prototxt')

maxItr = 5000
train_loss = np.zeros( maxItr )
test_loss = np.zeros( maxItr )
for it in range(maxItr):
    startTime = time.time()
    solver.step(1)

    train_loss[ it ] = solver.net.blobs['loss'].data
    # print "solver.net.blobs['data'].data.dtype", solver.net.blobs['data'].data.dtype
    # print "solver.net.blobs['label'].data.dtype", solver.net.blobs['label'].data.dtype
    # for i in range(5):
    #     cv2.imshow( 'win', solver.net.blobs['data'].data[i,:,:,:].transpose(1,2,0) )
    #     cv2.waitKey(0)
    #test_loss[ it ] = solver.test_nets[0].blobs['loss'].data
    if it%1000 == 0:
        np.save('train_loss.npy', train_loss[0:it])
        #np.save('test_loss.npy', test_loss[0:it])
    print 'iter=%d, time=%f, loss=%f' % (it, time.time() - startTime, train_loss[it])
