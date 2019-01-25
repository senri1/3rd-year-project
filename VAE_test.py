import tensorflow as tf
import os
import matplotlib as plt

test = collectData(10)

CAE = tf.keras.models.load_model( os.getcwd()+'/CAE.h5' )

tf.keras.models.load_model( os.getcwd()+'/encoder.h5' )

tf.keras.models.load_model( os.getcwd()+'/decoder.h5' )

print(test[0,:,:,0].shape)
print(CAE.predict(test[0:1,:,:,:])[0,:,:,0].shape )

plt.figure(0)
plt.imshow( test[0,:,:,3] )
plt.show()
plt.figure(1)
plt.imshow( CAE.predict(test[0:1,:,:,:])[0,:,:,3] )
plt.show()
