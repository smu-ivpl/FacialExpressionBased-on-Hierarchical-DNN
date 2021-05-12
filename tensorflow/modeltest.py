import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from util import getImageData,error_rate, init_weight_and_bias, y2indicator
from convert_jaffe_files import getImageDataJaffe
from convert_ckplus_files import getImageDatackplus
from ann_tf import HiddenLayer
import itertools
import os

#import brewer2mpl
#set3 = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors
# differences from Theano:
# image dimensions are expected to be: N x width x height x color
# filter shapes are expected to be: filter width x filter height x input feature maps x output feature maps


def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32)


class ConvPoolLayer(object):
    def __init__(self, mi, mo, fw=5, fh=5, poolsz=(2, 2)):
        # mi = input feature map size
        # mo = output feature map size
        sz = (fw, fh, mi, mo)
        #print(fw, fh)
        W0 = init_filter(sz, poolsz)
        self.W = tf.Variable(W0)
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = tf.Variable(b0)
        self.poolsz = poolsz
        self.params = [self.W, self.b]

    def forward(self, X):
        #print("forward시작..............")
       # print(X)
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
        #print(conv_out)
        conv_out = tf.nn.bias_add(conv_out, self.b)
        p1, p2 = self.poolsz
        pool_out = tf.nn.max_pool(
            conv_out,
            ksize=[1, p1, p2, 1],
            strides=[1, p1, p2, 1],
            padding='SAME'
        )
        #print(pool_out)
        return tf.tanh(pool_out)


class CNN(object):
    def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, lr=10e-4, mu=0.99, reg=10e-4, decay=0.99999, eps=10e-3, batch_sz=30, epochs=30, show_fig=True):
        lr = np.float32(lr)#learning rate
        mu = np.float32(mu)#momentum
        reg = np.float32(reg)#regularization
        decay = np.float32(decay)#decay
        eps = np.float32(eps)#epoch
        K = len(set(Y))#set Y 길이(emotion label들.. 총 몇개 image data갯수)
        X2 = X.astype(np.float32)#형변환. float32로 바꾼다
        Y2 = y2indicator(Y).astype(np.float32)
        Xv, Yv = X2[500:549], Y2[500:549]
        X1, Y1 = X2[500:549], Y2[500:549]
        # make a validation set
        X, Y = shuffle(X, Y)#csv파일의 image data들의 쌍들의 순서를 random으로 섞는다.
        X = X.astype(np.float32)#형변환. float32로 바꾼다
        Y = y2indicator(Y).astype(np.float32)
        
        Xvalid, Yvalid = X[-1000:], Y[-1000:]#앞에서부터 1000개부터 끝까지
        print('x의 길이:',len(Xvalid))
        X, Y = X[:-1000], Y[:-1000]#뒤에서 1000까지...
        print('x의 길이2:',len(Xvalid))
        Yvalid_flat = np.argmax(Yvalid, axis=1) # for calculating error rate
        
        # make a Test set
        #X, Y = shuffle(X, Y)#csv파일의 image data들의 쌍들의 순서를 random으로 섞는다.
        #X = X.astype(np.float32)#형변환. float32로 바꾼다
        #Y = y2indicator(Y).astype(np.float32)
        
        XTest, YTest = X[-1000:], Y[-1000:]#앞에서부터 1000개부터 끝까지
        #X, Y = X[:-100], Y[:-100]#뒤에서 1000까지...
        #YTest_flat = np.argmax(YTest, axis=1) # for calculating error rate
        
        # initialize convpool layers
        N, width, height, c = X.shape
        #print('xshape:',X.shape)
        #print('N: ',N)
        mi = c
        outw = width
        outh = height
        self.convpool_layers = []
        #print("convpool for문 시작")
        for mo, fw, fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi, mo, fw, fh)
            #print(layer.params)
            self.convpool_layers.append(layer)
            outw = outw / 2
            outh = outh / 2
            #print("ouput:",mo,"input:",mi)
            mi = mo
        #print(len(self.convpool_layers))
       # print(self.convpool_layers[0])
        # initialize mlp layers
        self.hidden_layers = []
        M1 = int(round(self.convpool_layer_sizes[-1][0]*outw*outh)) # size must be same as output of last convpool layer
        count = 0
       # print("hiddenlayer for문 시작")
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            #print(M1,M2,count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1

        # logistic regression layer
        W, b = init_weight_and_bias(M1, K)
        self.W = tf.Variable(W, 'W_logreg')
        self.b = tf.Variable(b, 'b_logreg')

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.convpool_layers:
            self.params += h.params
        for h in self.hidden_layers:
            self.params += h.params

        # set up tensorflow functions and variables
        tfX = tf.placeholder(tf.float32, shape=(None, width, height, c), name='X')
        tfY = tf.placeholder(tf.float32, shape=(None, K), name='Y')
        cfin=c
        
        act = self.forward(tfX)

        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=act,
                labels=tfY
            )
        ) + rcost
        #print("TFX의 형태: ",tfX)
        prediction = self.predict(tfX)
        train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)

        n_batches = int(round(N / batch_sz))#batch size 를 N개(SAMPLE 갯수?)로 나눈것.
        costs = []
        tcosts = []
        #prediction result##########################################################
        #X, Y = X[:-10], Y[:-10]#뒤에서 10까지...
        keX = tf.placeholder(tf.float32, shape=(None, width, height, cfin), name='X1')
        keY = tf.placeholder(tf.float32, shape=(None, K), name='Y1')
        #Xv, Yv = X1[0:36], Y1[0:36]        
        y_prob=[]
        prob_v=self.prob(keX)
        #prediction1 = self.predict(keX)
        y_pred=[]
        
        Yv_flat = np.argmax(Yv, axis=1)
        y_true = Yv_flat
        #print('y_true: ',y_true, len(y_true))
        accuracy=[]
        p=[]
        #########################################################################
        saver = tf.train.Saver()
        
        with tf.Session() as session:
            saver.restore(session,'C:/FacialExpression/facialExpression/tensorflow/facialParam.ckpt')
            #session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):#N 개를 SAMPLE로 나눈것만큼 돈다...
                    Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                    Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                    session.run(train_op, feed_dict={tfX: Xbatch, tfY: Ybatch})
                    
                    if j % 20 == 0:
                        c = session.run(cost, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        c2 = session.run(cost, feed_dict={tfX: XTest, tfY: YTest})
                        costs.append(c)
                        tcosts.append(c2)
                        
                        p = session.run(prediction, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        e = error_rate(Yvalid_flat, p)
                        py_prob=session.run(prob_v, feed_dict={keX: Xv, keY: Yv})
                        min_max_scaler = preprocessing.MinMaxScaler()
                        normalized = min_max_scaler.fit_transform(py_prob)
                        y_prob=normalized
                        #print('y_prob',y_prob)
                        y_pred = [np.argmax(prob) for prob in y_prob]
                        #print('y_pred',y_pred)
                        #print(p)
                        a = accuracy_score(Yvalid_flat, p)
                        accuracy.append(a)
                        print( "i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)
                        #print(i, np.mean(YTest_flat)==session.run(prediction, feed_dict={tfX: XTest, tfY: YTest}))
            saver = tf.train.Saver()
            save_path = saver.save(session, 'C:/FacialExpression/facialExpression/tensorflow/facialParam1.ckpt')
            print (os.getcwd())
            print("Model saved in file: ", save_path)
        
        if show_fig:
        #test##########################################################
            legend1, = plt.plot(costs,label="train cost")
            legend2, = plt.plot(tcosts,label="test cost")
            plt.legend([legend1, legend2])
            plt.show()
            legend1, = plt.plot(accuracy,label="accuracy")
            plt.legend([legend1])
            plt.show()
            cm=confusion_matrix(Yvalid_flat, p)
            cmap=plt.cm.Blues
            title='Confusion matrix'
            classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, int(cm[i, j]*100)/100.0,
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            #plt.show()
            #self.plot_confusion_matrix(cnf_matrix,normalize=False)
            plt.show()
            self.plot_subjects_with_probs(0, 49, y_prob, y_pred, y_true, X1)
        #########################################################################
    
        #########################################################################
    def forward(self, X):
        Z = X
        for c in self.convpool_layers:
            Z = c.forward(Z)
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        pY = self.forward(X)
        return tf.argmax(pY, 1)
    
    def prob(self, X):
        pY = self.forward(X)
        return pY
    #########################################################################
    def plot_subjects(self, start, end, y_pred, y_true, X, title=False):
        fig = plt.figure(figsize=(12,12))
        emotion = {0:'Angry', 1:'Disgust',2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
        for i in range(int(start), int(end)+1):
            #input_img = X[i:(i+1),:]
            #input_img = X[i].reshape(48, 48)
            input_img = X[i].reshape(128, 128)
            ax = fig.add_subplot(7,7,i+1)
            ax.imshow(input_img, cmap=matplotlib.cm.gray)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            if y_pred[i] != y_true[i]:
                #plt.xlabel(emotion[y_true[i]], color='#53b3cb',fontsize=12)
                plt.xlabel(emotion[y_true[i]], color='blue',fontsize=12)
                #print(y_true[i])
            else:
                plt.xlabel(emotion[y_true[i]], fontsize=12)
                #print(y_true[i])
            if title:
                plt.title(emotion[y_pred[i]], color='blue')
            plt.tight_layout()
        plt.show()
            
    def plot_probs(self, start,end, y_prob, X):
        fig = plt.figure(figsize=(12,12))
        for i in range(int(start), int(end)+1):
            #input_img = X[i:(i+1),:,:,:]
            ax = fig.add_subplot(7,7,i+1)
            ax.bar(np.arange(0,7), y_prob[i], alpha=0.5)
            ax.set_xticks(np.arange(0.5,7.5,1))
            labels = ['Angry', 'Disgust','Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            ax.set_xticklabels(labels, rotation=90, fontsize=10)
            ax.set_yticks(np.arange(0.0,1.1,0.5))
            plt.tight_layout()
        plt.show()
        
    def plot_subjects_with_probs(self, start, end, y_prob, y_pred, y_true, X):
        iter = int(end - start)/7
        for i in np.arange(0,iter):
            self.plot_subjects(i*7,(i+1)*7-1, y_pred, y_true, X, title=False)
            self.plot_probs(i*7,(i+1)*7-1, y_prob,X)

    
def main():
    #X, Y = getImageDataJaffe()
    #X, Y = getImageData()
    #X1, Y1 = getImageData()
    X2, Y2 = getImageDataJaffe()
    X3, Y3 = getImageDatackplus()
    X=np.concatenate((X2, X3), axis=0)
    X=np.concatenate((X, X2), axis=0)
    Y=np.concatenate((Y2, Y3), axis=0)
    Y=np.concatenate((Y, Y2), axis=0)
    # reshape X for tf: N x w x h x c
    X, Y = shuffle(X, Y)
    X = X.transpose((0, 2, 3, 1))
    #print( "X.shape:", X.shape)

    model = CNN(
        convpool_layer_sizes=[(64, 5, 5), (64, 5, 5)],
        hidden_layer_sizes=[500, 300],
    )
    model.fit(X, Y)
    
    #pre
    #model.test(pre.p)
if __name__ == '__main__':
    main()
