import caffe
import cv2
import time
import numpy as np


def loadImages( sub_list ):
    imStack = np.zeros( (len(sub_list),3,28,28) )
    labelStack = np.zeros( (len(sub_list),2) )

    c=0
    for ele in sub_list:
        sp = ele.split(' ')
        #print 'Read Image : ', sp[0], 'label : ', sp[1]
        im = cv2.imread( sp[0] )
        imX = im.astype('float32') /255.0 - 0.5 #having number (-0.5, 0.5)
        imStack[c,:,:,:] = imX.transpose(2,0,1)
        labelStack[c,0] = int(sp[1])
        labelStack[c,1] = int(sp[1])+1
        c = c + 1
        #print im.shape
        #cv2.imshow( 'win', im )
        #cv2.waitKey(0)

    return imStack, labelStack


with open("train/annotation.txt") as f:
    mylist = f.read().splitlines()


caffe.set_mode_gpu()
caffe.set_device(0)

solver = None
solver = caffe.SGDSolver('solver_im.prototxt')

maxItr = 5000
train_loss = np.zeros( maxItr )
test_loss = np.zeros( maxItr )
for it in range(maxItr):
    startTime = time.time()
    ip = it % 100
    print 'read ', 100*ip, ' to ', 100*ip+100
    imStack, labelStack = loadImages( mylist[100*ip:100*ip+100] )
    # print 'imStack.dtype', imStack.astype('float32').dtype
    # print 'labelStack.dtype', labelStack.astype('float32').dtype
    # print "solver.net.blobs['data'].data.dtype", solver.net.blobs['data'].data.dtype
    # print "solver.net.blobs['label'].data.dtype", solver.net.blobs['label'].data.dtype
    solver.net.blobs['data'].data[:,:,:,:] = imStack[:,:,:,:]#.astype('float32')
    solver.net.blobs['label'].data[:,:] = labelStack[:,:]#.astype('float32')
    # for i in range(5):
    #     cv2.imshow( 'win', solver.net.blobs['data'].data[i,:,:,:].transpose(1,2,0) )
    #     cv2.waitKey(0)
    solver.step(1)

    train_loss[ it ] = solver.net.blobs['loss'].data
    #test_loss[ it ] = solver.test_nets[0].blobs['loss'].data
    if it%1000 == 0:
        np.save('train_loss_im.npy', train_loss[0:it])
        #np.save('test_loss_im.npy', test_loss[0:it])
    print 'iter=%d, time=%f, loss=%f' % (it, time.time() - startTime, train_loss[it])
