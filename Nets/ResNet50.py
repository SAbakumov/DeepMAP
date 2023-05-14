import tensorflow as tf


def res_conv(x, s, filters):

  x_skip = x
  f1, f2 = filters

  # first block
  x = tf.keras.layers.Conv1D(f1, kernel_size=1, strides=s, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)


  # second block
  x = tf.keras.layers.Conv1D(f1, kernel_size=1, strides=1, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)

  #third block
  x = tf.keras.layers.Conv1D(f2, kernel_size=1, strides=1, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  # shortcut 
  x_skip = tf.keras.layers.Conv1D(f2, kernel_size=1, strides=s, padding='valid')(x_skip)
  x_skip = tf.keras.layers.BatchNormalization()(x_skip)

  # add 
  x = tf.keras.layers.Add()([x, x_skip])
  x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)

  return x




def res_identity(x, filters): 
 

  x_skip = x # this will be used for addition with the residual block 
  f1, f2 = filters

  #first block 
  x = tf.keras.layers.Conv1D(f1, kernel_size=1, strides=1, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)

  #second block # bottleneck (but size kept same with padding)
  x = tf.keras.layers.Conv1D(f1, kernel_size=3, strides=1, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)

  # third block activation used after adding the input
  x = tf.keras.layers.Conv1D(f2, kernel_size=1, strides=1, padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)

  # add the input 
  x = tf.keras.layers.Add()([x, x_skip])
  x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)

  return x



def resnet50(input_shape):

  input_im =  tf.keras.layers.Input(shape = input_shape) # cifar 10 images size


  x = tf.keras.layers.Conv1D(64, kernel_size=7, strides=2)(input_im)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)
  x = tf.keras.layers.MaxPooling1D(3, strides=2)(x)

  #2nd stage 
  # frm here on only conv block and identity block, no pooling

  x = res_conv(x, s=1, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = res_conv(x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))

  # 4th stage

  x = res_conv(x, s=2, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))

  # 5th stage

  x = res_conv(x, s=2, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))

  # ends with average pooling and dense connection

  x = tf.keras.layers.GlobalAveragePooling1D()(x)

  x = tf.keras.layers.Dense(units=256, activation = 'elu')(x)
  x = tf.keras.layers.Dense(units=128, activation = 'elu')(x)

  output = tf.keras.layers.Dense(units=1, activation = 'sigmoid')(x)

  # define the model 

  model = tf.keras.Model(inputs=input_im, outputs=output, name='Resnet50')

  return model

if __name__ == '__main__':
  model = resnet50()
  model.summary()
