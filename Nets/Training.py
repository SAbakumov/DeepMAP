import tensorflow as tf, os,csv

class SaveMetrics(tf.keras.callbacks.Callback):
    def __init__(self,save_path):
        self.save_path = save_path
        with open(os.path.join(self.save_path,'history.csv'),'w', newline='') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(['Epoch','Train loss','Val loss'])
            
    def on_epoch_end(self, epoch, logs=None):
        val_l = logs['val_loss']
        train_l = logs['loss']
        with open(os.path.join(self.save_path,'history.csv'),'a', newline='') as csvfile:
            w = csv.writer(csvfile)
            w.writerow([epoch,train_l,val_l])



class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self,save_path, save_best_metric='val_loss'):
        self.save_best_metric = save_best_metric
        self.best = float('inf')
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if metric_value < self.best:
            self.best = metric_value
            self.model.save_weights(os.path.join(self.save_path,'modelBestLoss.hdf5' ))


class TrainModel(tf.keras.Model):
    def __init__(self,mdl,**kwargs):
        super(TrainModel, self).__init__(**kwargs)

        self.bce_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        self.model = mdl


    def train_step(self, data):
        with tf.GradientTape() as tape:
            outputs = self.model(data[0][0])
            bce_loss = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(data[0][1],outputs),axis=-1)
           
        gradients = tape.gradient(bce_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.bce_loss.update_state(bce_loss)
        return {"loss": self.bce_loss.result()}

    def test_step(self, data):

        outputs = self.model(data[0])
        bce_loss_vals = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(data[1],outputs),axis=-1)
        self.val_loss.update_state(bce_loss_vals)

      
        return {"loss": self.val_loss.result()}





