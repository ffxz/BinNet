
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.platform import flags
import datasets
from data_generator import DataGenerator
import os
import time
from spectrogram_to_wave import recover_wav



flags.DEFINE_string('task', 'libri', 'set task name of this program')
flags.DEFINE_string('train_dataset', 'train-clean-100', 'set the training dataset')
flags.DEFINE_string('mode', 'infer', 'choice =[train, test, infer]')
flags.DEFINE_string('feature_mode', 'magnitude', 'choice =[magnitude, complex]')
flags.DEFINE_string('workspace', 'G:\\研二上学期\\paper\\denoise\\coding\\BinNet', 'set dir to your root')
flags.DEFINE_string('speech_dir', 'data\\clean', 'set dir to your speech')
flags.DEFINE_string('noise_dir', 'data\\noise', 'set dir to your noise')
flags.DEFINE_integer('snr', 5, 'snr')
flags.DEFINE_integer('n_hop', 4, 'frame hop length')
flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_integer('fs', 8000, 'fs')
flags.DEFINE_integer('n_concat', 8, 'frame length')
flags.DEFINE_integer('magnification', 1, 'feature type')
flags.DEFINE_integer('sample_rate', 16000, 'sample_rate')
flags.DEFINE_integer('n_window', 512, 'n_window')
flags.DEFINE_integer('n_overlap', 256, 'n_overlap')

flags.DEFINE_float('te_snr', 5, 'te_snr')
flags.DEFINE_float('tr_snr', 5, 'tr_snr')

FLAGS = flags.FLAGS

def train():
    #first step: prepare data for training
    print('start to prepare data')
    datasets.processing(FLAGS)
    workspace = FLAGS.workspace
    tr_snr = FLAGS.tr_snr
    #datasets.create_mixture_csv(FLAGS)
    #datasets.calculate_mixture_features(FLAGS)
    #datasets.pack_features(FLAGS)
    #input_data, label = datasets.processing(FLAGS)
    tr_hdf5_path = os.path.join(workspace, "datasave", "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "data.h5")
    (tr_x, tr_y) = datasets.load_hdf5(tr_hdf5_path)

    
    batch_size = 3
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))
    (_, n_concat, n_freq) = tr_x.shape
    
    #second step: adding model
    
    #model = Model()
    inputs = tf.keras.Input(shape=(n_concat, n_freq))  # Returns a placeholder tensor
    inputs1 = layers.Flatten()(inputs)

    # A layer instance is callable on a tensor, and returns a tensor.
    x = layers.Dense(512, activation='relu')(inputs1)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(n_freq, activation='linear')(x)
    predictions = layers.Reshape((1, n_freq), name='ctg_out_1')(predictions)
    predictions1 = layers.Dense(n_freq, activation='linear')(x)
    predictions1 = layers.Reshape((1, n_freq), name='ctg_out_2')(predictions1)
    model = tf.keras.Model(inputs=inputs, outputs=[predictions, predictions1])

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss={'ctg_out_1':'mse',
              'ctg_out_2':'mse'},
              metrics=['accuracy'])

    model.summary()   

    t1 = time.time()
    iter = 0
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        print('hello')
        loss = model.train_on_batch(batch_x, [batch_y, batch_y])
        #model.fit(batch_x, [batch_y, batch_y], batch_size=50, epochs=5000)
        iter += 1

        
        if iter == 10:
            break

    print("Training time: %s s" % (time.time() - t1,))    
    
    # Trains for 5 epochs
    #model.fit(data, labels, batch_size=50, epochs=5000)
    
    #third step: fit data to model
    #model.fit(input_data, label, batch_size=50, epochs=5000)
    
    #forth step: save model to local

    
def train_bin():
    #data_x, data_y = datasets.get_data(FLAGS)
    (tr_x, tr_y) = datasets.get_data(FLAGS)
    tr_x = np.array(tr_x)  # (n_segs, n_concat, n_freq)
    tr_y = np.array(tr_y)
    tr_y = tr_y[:, np.newaxis, :]  # (n_segs, n_freq) => (n_segs, 1 ,n_freq)
    batch_size = 3
    (_, n_concat, n_freq) = tr_x.shape
    # second step: adding model
    # model = Model()
    inputs = tf.keras.Input(shape=(n_concat, n_freq))  # Returns a placeholder tensor
    inputs1 = layers.Flatten()(inputs)

    # A layer instance is callable on a tensor, and returns a tensor.
    x = layers.Dense(512, activation='relu')(inputs1)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(n_freq, activation='linear')(x)
    predictions = layers.Reshape((1, n_freq), name='bin_out_1')(predictions)
    predictions1 = layers.Dense(n_freq, activation='linear')(x)
    predictions1 = layers.Reshape((1, n_freq), name='bin_out_2')(predictions1)
    model = tf.keras.Model(inputs=inputs, outputs=[predictions, predictions1])

    checkpoint_path = os.path.join(FLAGS.workspace, 'log\\ck.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss={'bin_out_1': 'mse',
                        'bin_out_2': 'mse'},
                  metrics=['accuracy'])

    model.summary()
    model.fit(tr_x, [tr_y, tr_y], batch_size=5, epochs=2, callbacks=[cp_callback, tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir)])
    #first step: prepare data for testing

    #second step: restoring model 

    #third step: fit data to model

    #forth step: calculate error or result

    print('start to evaluate')
    loss, loss1, acc, acc1, q = model.evaluate(tr_x, [tr_y, tr_y])
    print('loss:%f' %loss, 'acc%f' %acc, loss1, acc1)

def test():
    pass
    workspace = FLAGS.workspace
    tr_snr = FLAGS.tr_snr
    (tr_x, tr_y) = datasets.get_data(FLAGS)
    tr_x = np.array(tr_x)  # (n_segs, n_concat, n_freq)
    tr_y = np.array(tr_y)
    tr_y = tr_y[:, np.newaxis, :]  # (n_segs, n_freq)    => (n_segs, 1 ,n_freq)

    batch_size = 3

    (_, n_concat, n_freq) = tr_x.shape

    checkpoint_path = os.path.join(FLAGS.workspace, 'log\\ck.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    inputs = tf.keras.Input(shape=(n_concat, n_freq))  # Returns a placeholder tensor
    inputs1 = layers.Flatten()(inputs)

    # A layer instance is callable on a tensor, and returns a tensor.
    x = layers.Dense(512, activation='relu')(inputs1)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(n_freq, activation='linear')(x)
    predictions = layers.Reshape((1, n_freq), name='bin_out_1')(predictions)
    predictions1 = layers.Dense(n_freq, activation='linear')(x)
    predictions1 = layers.Reshape((1, n_freq), name='bin_out_2')(predictions1)
    model = tf.keras.Model(inputs=inputs, outputs=[predictions, predictions1])
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss={'bin_out_1': 'mse',
                        'bin_out_2': 'mse'},
                  metrics=['accuracy'])

    model.load_weights(latest)
    print('start to infer')
    loss, loss1, acc, acc1, q = model.evaluate(tr_x, [tr_y, tr_y])
    print('loss:%f' % loss, 'acc%f' % acc, loss1, acc1)

    a = model.predict(tr_x)
    print(a)

def infer():
    workspace = FLAGS.workspace
    tr_snr = FLAGS.tr_snr
    tr_x, mixed_complx_x = datasets.get_data_infer(FLAGS)
    tr_x = np.array(tr_x)  # (n_segs, n_concat, n_freq)

    batch_size = 3

    (_, n_concat, n_freq) = tr_x.shape

    checkpoint_path = os.path.join(FLAGS.workspace, 'log\\ck.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    inputs = tf.keras.Input(shape=(n_concat, n_freq))  # Returns a placeholder tensor
    inputs1 = layers.Flatten()(inputs)

    # A layer instance is callable on a tensor, and returns a tensor.
    x = layers.Dense(512, activation='relu')(inputs1)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(n_freq, activation='linear')(x)
    predictions = layers.Reshape((1, n_freq), name='bin_out_1')(predictions)
    predictions1 = layers.Dense(n_freq, activation='linear')(x)
    predictions1 = layers.Reshape((1, n_freq), name='bin_out_2')(predictions1)
    model = tf.keras.Model(inputs=inputs, outputs=[predictions, predictions1])
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss={'bin_out_1': 'mse',
                        'bin_out_2': 'mse'},
                  metrics=['accuracy'])

    model.load_weights(latest)
    print('start to infer')

    pred_real, pred_img = model.predict(tr_x)
    # Recover enhanced wav.
    pred_sp = np.exp(pred_real)
    t1, t2, t3 = pred_sp.shape
    dd = pred_sp.reshape(t1, t3)

    s = recover_wav(mixed_complx_x, mixed_complx_x, FLAGS.n_overlap, np.hamming)
    s *= np.sqrt((np.hamming(FLAGS.n_window) ** 2).sum())  # Scaler for compensate the amplitude
    # change after spectrogram and IFFT.

    # Write out enhanced wav.
    out_path = "tt.enh.wav"
    #datasets.create_folder(os.path.dirname(out_path))
    datasets.write_audio(out_path, s, 8000)



if __name__ == '__main__':
    if FLAGS.mode == 'train':
        train_bin()
    if FLAGS.mode == 'test':
        test()
    if FLAGS.mode == 'infer':
        infer()
    
   




