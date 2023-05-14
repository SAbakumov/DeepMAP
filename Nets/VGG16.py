import tensorflow as tf, os,csv






def  VGG16(input_shape):
    input = tf.keras.layers.Input(shape = input_shape)

    x = tf.keras.layers.Conv1D(64,3,padding='same',activation='elu')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(128,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(256,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)


    x = tf.keras.layers.Conv1D(512,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(512,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(512,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(512,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(512,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(512,3,padding='same',activation='elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)


    x = tf.keras.layers.Dense(units=256, activation = 'elu')(x)
    x = tf.keras.layers.Dense(units=128, activation = 'elu')(x)

    output = tf.keras.layers.Dense(units=1, activation = 'sigmoid')(x)
    
    CNN = tf.keras.Model(input,output)
    return CNN

if __name__ == '__main__':

    model = VGG16()
    model.summary()
