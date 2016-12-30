import numpy as np
import matplotlib.pyplot as plt

train_loss = np.load( 'train_loss_im.npy' )

window = 10
smooth_train_loss = np.convolve(train_loss[0:], np.ones( window )/window, 'valid' )

plt.plot( train_loss[0:], 'r-', smooth_train_loss, 'b' )
plt.figure()
plt.plot( np.log(smooth_train_loss[0:] ) , 'r-' )
plt.show()



#test_loss = np.load( 'test_loss.npy' )
#smooth_test_loss = np.convolve(test_loss[0:], np.ones( 5*window )/(5*window), 'valid' )
# fig, ax1 = plt.subplots()
# ax1.plot( train_loss[50:], 'r-', smooth_train_loss, 'b' )
# ax2 = ax1.twinx()
# ax2.plot( test_loss[50:], 'g-' )
#
# fig, ax_log = plt.subplots()
# ax_log.plot( np.log(smooth_train_loss[0:] ) , 'r-' )
# #ax_log.plot( np.log(smooth_test_loss[0:] ) , 'g-' )
# ax_log.plot( np.log(test_loss[0:] ) , 'g-' )
#
# plt.show()
# #fig.savefig( 'b.png')
