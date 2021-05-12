#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.utils import np_utils
from cnn_keras import CNN
from keras.models import load_model
import itertools
from itertools import combinations
import get_dataset as getdb
import convert_ckplus_files as ck
import convert_jaffe_files as jaffe
import convert_ferg_files as ferg
import convert_sunglass_files as sunglass
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

'''def loading_face_data():
	X1, Y1 = ferg.getDataFERG()
	X2, Y2 = jaffe.getDataJaffe()
	X3, Y3 = ck.getDatackplus()

	X=np.concatenate((X2, X3), axis=0)
	X=np.concatenate((X, X2), axis=0)
	X=np.concatenate((X, X1), axis=0)

	Y=np.concatenate((Y2, Y3), axis=0)
	Y=np.concatenate((Y, Y2), axis=0)
	Y=np.concatenate((Y, Y1), axis=0)
	# reshape X for th: N x w x h x c
	X, Y = shuffle(X,Y)
	print(len(X))
	print('==loading image data==')

	return X, Y'''

def loading_face_data():
    X=[]
    Y=[]
    X,Y=getdb.get_jaffe_path(X,Y)
    X2, Y2 = jaffe.getDataJaffe()
    sun_X = sunglass.getDataSunglass()

    for i in range(len(Y)):
        if os.path.basename(X[i])[:2] == 'KA':
            sun_X=np.concatenate((sun_X,[X2[i]]),axis=0)
    for j in range(0,5):
        sun_X = np.concatenate((sun_X, sun_X), axis=0)
    print(len(sun_X))
    print('==loading image data==')
    sun_X = shuffle(sun_X)
    return sun_X

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

#--------------------------------

X = loading_face_data()
X = shuffle(X)

N, D = X.shape
#K = len(set(Y))
#print("y길이",len(Y))
#tfX = tf.placeholder(tf.float32, shape=(None, width, height, t), name='X')
tfX = tf.placeholder(tf.float32, shape=[None, 128*128], name='X')
#tfY = tf.placeholder(tf.float32, shape=(None, K), name='Y')
# reshape X for tf: N x w x h x c
#X, Y = shuffle(X,Y)
#--------------------------------


D_W1 = tf.Variable(xavier_init([128*128, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


#Z = tf.placeholder(tf.float32, shape=[None, 16])
#c = tf.placeholder(tf.float32, shape=[None, 10])
Z = tf.placeholder(tf.float32, shape=[None, 100])
c = tf.placeholder(tf.float32, shape=[None, 4])

#G_W1 = tf.Variable(xavier_init([26, 256]))
G_W1 = tf.Variable(xavier_init([107, 256]))
G_b1 = tf.Variable(tf.zeros(shape=[256]))

G_W2 = tf.Variable(xavier_init([256, 128*128]))
G_b2 = tf.Variable(tf.zeros(shape=[128*128]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


Q_W1 = tf.Variable(xavier_init([128*128, 128]))
Q_b1 = tf.Variable(tf.zeros(shape=[128]))

Q_W2 = tf.Variable(xavier_init([128, 4]))
Q_b2 = tf.Variable(tf.zeros(shape=[4]))

#Q_W2 = tf.Variable(xavier_init([128, 10]))
#Q_b2 = tf.Variable(tf.zeros(shape=[10]))

theta_Q = [Q_W1, Q_W2, Q_b1, Q_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def sample_c(m):
    return np.random.multinomial(1, 4*[0.1], size=m)

def generator(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob


def Q(x):
    Q_h1 = tf.nn.relu(tf.matmul(x, Q_W1) + Q_b1)
    Q_prob = tf.nn.softmax(tf.matmul(Q_h1, Q_W2) + Q_b2)

    return Q_prob


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(128, 128), cmap='Greys_r')

    return fig


G_sample = generator(Z, c)
D_real = discriminator(tfX)
D_fake = discriminator(G_sample)
Q_c_given_x = Q(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real + 1e-8) + tf.log(1 - D_fake + 1e-8))
G_loss = -tf.reduce_mean(tf.log(D_fake + 1e-8))

cross_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(Q_c_given_x + 1e-8) * c, 1))
ent = tf.reduce_mean(-tf.reduce_sum(tf.log(c + 1e-8) * c, 1))
Q_loss = cross_ent + ent

D_solver = tf.train.AdamOptimizer(0.00001).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(0.0001).minimize(G_loss, var_list=theta_G)
Q_solver = tf.train.AdamOptimizer(0.0001).minimize(Q_loss, var_list=theta_G + theta_Q)

mb_size = 20
Z_dim = 100



#mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
if not os.path.exists('infogan/model/my_test_model-99-2901.meta'):
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
else:
    saver = tf.train.import_meta_graph('infogan/model/my_test_model-99-2901.meta')
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('infogan/model/'))
    sess.run(tf.global_variables_initializer())

if not os.path.exists('infogan/out/'):
    os.makedirs('infogan/out/')

#i = 0
count = 0
n_batches = int(round(N / mb_size))#batch size 를 N개(SAMPLE 갯수)로 나눈것.
#j = 0
loss_to_draw1 = []
loss_to_draw2 = []
for i in range(10000):
    X = shuffle(X)
    loss_to_draw_epoch1 = []
    loss_to_draw_epoch2 = []
    for j in range(n_batches):#N 개를 SAMPLE로 나눈것만큼 돈다.
        if j % 100 == 0:
            #Z_noise = sample_Z(16, Z_dim)
            Z_noise = sample_Z(16, Z_dim)
            #idx = np.random.randint(0, 10)
            idx = np.random.randint(0, 4)
            #c_noise = np.zeros([16, 10])
            #c_noise[range(16), idx] = 1
            #c_noise = np.zeros([16, 10])
            c_noise = np.zeros([16, 4])
            c_noise[range(16), idx] = 1
            samples = sess.run(G_sample,
                               feed_dict={Z: Z_noise, c: c_noise})
    
            fig = plot(samples)
            filename=str(i).zfill(3)+"_"+str(count).zfill(3)
            plt.savefig('infogan/out/{}.png'.format(filename), bbox_inches='tight')
            count += 1
            plt.close(fig)
    
        #X_mb, _ = mnist.train.next_batch(mb_size)
        X_mb = X[j*mb_size:(j*mb_size+mb_size)]
        #_ = Y[j*mb_size:(j*mb_size+mb_size)]
        
        '''if j <= n_batches:
            j += 1
        else:
            j = 0'''
        #j += 1
            
        Z_noise = sample_Z(mb_size, Z_dim)
        c_noise = sample_c(mb_size)
    
        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={tfX: X_mb, Z: Z_noise, c: c_noise})
        loss_to_draw_epoch1.append(D_loss_curr)
        _, G_loss_curr = sess.run([G_solver, G_loss],
                                  feed_dict={Z: Z_noise, c: c_noise})
        loss_to_draw_epoch2.append(G_loss_curr)
        sess.run([Q_solver], feed_dict={Z: Z_noise, c: c_noise})
    
        if j % 100 == 0:
            print('epoch: {}'.format(i))
            print('j: {}'.format(j))
            print('nb: {}'.format(n_batches))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            saver.save(sess, 'infogan/model/my_test_model-{0}-{1}'.format((i),(n_batches)))
            # draw loss curve every epoch
            loss_to_draw1.append(np.mean(loss_to_draw_epoch1))
            loss_to_draw2.append(np.mean(loss_to_draw_epoch2))
            plt_save_dir = "infogan/graph/"
            plt_save_img_name = str(i) + '.png'
            legend1, = plt.plot(loss_to_draw1, label='d_loss',color='g')
            legend2, = plt.plot(loss_to_draw2, label='g_loss',color='r')
            plt.legend([legend1, legend2])
            plt.grid(True)
            plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))

'''for it in range(100):
    if it % 100 == 0:
        #Z_noise = sample_Z(16, Z_dim)
        Z_noise = sample_Z(16, Z_dim)
        #idx = np.random.randint(0, 10)
        idx = np.random.randint(0, 7)
        #c_noise = np.zeros([16, 10])
        #c_noise[range(16), idx] = 1
        #c_noise = np.zeros([16, 10])
        c_noise = np.zeros([16, 7])
        c_noise[range(16), idx] = 1
        samples = sess.run(G_sample,
                           feed_dict={Z: Z_noise, c: c_noise})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    #X_mb, _ = mnist.train.next_batch(mb_size)
    X_mb = X[j*mb_size:(j*mb_size+mb_size)]
    _ = Y[j*mb_size:(j*mb_size+mb_size)]
    
    j += 1
        
    Z_noise = sample_Z(mb_size, Z_dim)
    c_noise = sample_c(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss],
                              feed_dict={tfX: X_mb, Z: Z_noise, c: c_noise})

    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={Z: Z_noise, c: c_noise})

    sess.run([Q_solver], feed_dict={Z: Z_noise, c: c_noise})

    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()'''
