
import os
import math
import tensorflow as tf



import cv2
import numpy as np
import random

from data_generator import get_batch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Train_char_seg(object):
    def __init__(self):
        self.input_shape = (2048,64, 3)

        self.learningrate=0.0001
        self.epochs=20
        self.batch_size=16
        self.save_steps=100
        self.one_epoch_num=10000

        self.alpha=0.9
        self.beta=0.1

    def network(self):
        network = {}
        network["inputs"] = tf.placeholder(tf.float32, [self.batch_size, self.input_shape[1],self.input_shape[0], self.input_shape[2]],name='inputs')
        network["down-conv1"] = tf.layers.conv2d(inputs=network["inputs"], filters=32, kernel_size=(2, 2), padding="same",activation=tf.nn.relu, name="down-conv1")
        network["down-pool1"] = tf.layers.max_pooling2d(inputs=network["down-conv1"], pool_size=[2, 2], strides=2)
        network["down-conv2"] = tf.layers.conv2d(inputs=network["down-pool1"], filters=64, kernel_size=(2, 2), padding="same",activation=tf.nn.relu, name="down-conv2")
        network["down-pool2"] = tf.layers.max_pooling2d(inputs=network["down-conv2"], pool_size=[2, 2], strides=2)
        network["down-conv3"] = tf.layers.conv2d(inputs=network["down-pool2"], filters=128, kernel_size=(2, 2), padding="same",activation=tf.nn.relu, name="down-conv3")
        network["down-pool3"] = tf.layers.max_pooling2d(inputs=network["down-conv3"], pool_size=[2, 2], strides=2)
        network["down-conv4"] = tf.layers.conv2d(inputs=network["down-pool3"], filters=256, kernel_size=(2, 2), padding="same",activation=tf.nn.relu, name="down-conv4")
        network["down-pool4"] = tf.layers.max_pooling2d(inputs=network["down-conv4"], pool_size=[2, 2], strides=2)
        network["down-conv5"] = tf.layers.conv2d(inputs=network["down-pool4"], filters=512, kernel_size=(2, 2), padding="same",activation=tf.nn.relu, name="down-conv5")
        network["down-pool5"] = tf.layers.max_pooling2d(inputs=network["down-conv5"], pool_size=[2, 2], strides=2)
        network["down-conv6"] = tf.layers.conv2d(inputs=network["down-pool5"], filters=512, kernel_size=(2, 2), padding="same",activation=tf.nn.relu, name="down-conv6")
        network["down-pool6"] = tf.layers.max_pooling2d(inputs=network["down-conv6"], pool_size=[2, 2], strides=2)
        
        network["up-conv1"] = tf.layers.conv2d_transpose(inputs=network["down-pool6"], filters=512, kernel_size=(1, 2),strides=(1, 2), padding="valid",activation=tf.nn.relu, name="up-conv1")
        network["up-conv2"] = tf.layers.conv2d_transpose(inputs=network["up-conv1"], filters=512, kernel_size=(1, 2),strides=(1, 2), padding="valid",activation=tf.nn.relu, name="up-conv2")
        network["up-conv3"] = tf.layers.conv2d_transpose(inputs=network["up-conv2"], filters=256, kernel_size=(1, 2),strides=(1, 2), padding="valid",activation=tf.nn.relu, name="up-conv3")
        network["up-conv4"] = tf.layers.conv2d_transpose(inputs=network["up-conv3"], filters=128, kernel_size=(1, 2),strides=(1, 2), padding="valid",activation=tf.nn.relu, name="up-conv4")
        network["up-conv5"] = tf.layers.conv2d_transpose(inputs=network["up-conv4"], filters=64, kernel_size=(1, 2), strides=(1, 2),padding="valid",activation=tf.nn.relu, name="up-conv5")
        network["up-conv6"] = tf.layers.conv2d_transpose(inputs=network["up-conv5"], filters=1, kernel_size=(1, 2), strides=(1, 2),padding="valid",activation=None, name="up-conv6")
        
        network["outputs"] =tf.contrib.layers.flatten(network["up-conv6"])
        return network

    def Dynamic_Weighted_Binary_CrossEntropy_loss(self,y_true, y_pred, alpha, beta):
        """
        Heuristic Rules for Dynamic Loss
        L(p; q) = −α Xi;qi=1log pi − β Xi;qi=0log(1 − pi)
        accpos = Xi;qi=11(pi > 0:5)= Xi;qi=11;
        accneg = Xi;qi=01(pi < 0:5)= Xi;qi=01:
        """
        y_pred=tf.nn.sigmoid(y_pred)
        y_pred=tf.cast(y_pred>0.5,tf.float32)
        
        acc_pos_fenzi = tf.reduce_sum(tf.multiply(y_pred, y_true))
        acc_pos_fenmu = tf.reduce_sum(y_true)
        acc_neg_fenzi = tf.reduce_sum(tf.multiply((1.0 - y_pred), (1.0 - y_true)))
        acc_neg_fenmu = tf.reduce_sum(1.0 - y_true)
        acc_pos = tf.div(acc_pos_fenzi ,tf.add(acc_pos_fenmu , 1e-10))
        acc_neg = tf.div(acc_neg_fenzi ,tf.add(acc_neg_fenmu , 1e-10))
        seigema = tf.minimum(beta, 0.001)

        alpha_new=tf.where(tf.less(acc_pos ,acc_neg),tf.add(alpha,seigema),tf.subtract(alpha , seigema))
        alpha_op=tf.assign(alpha,alpha_new)
        beta_new=tf.where(tf.less(acc_pos ,acc_neg),tf.subtract(beta ,seigema),tf.add(beta , seigema))
        beta_op=tf.assign(beta,beta_new)
        

        accuracy = tf.div(tf.add(acc_pos_fenzi , acc_neg_fenzi), tf.add(acc_pos_fenmu , tf.add(acc_neg_fenmu , 1e-10)))
        return alpha, beta, accuracy,alpha_op,beta_op
    
    

    def train(self):
        #network
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=self.learningrate,
                                               global_step=global_step,
                                               decay_steps=2000,
                                               decay_rate=0.1,
                                               staircase=True)
 
        model=self.network()
        alpha =tf.Variable(self.alpha, trainable=False)
        beta =tf.Variable(self.beta, trainable=False)
        labels = tf.placeholder(tf.float32, [self.batch_size, self.input_shape[0]], name='labels')
        
        loss = -tf.reduce_mean(alpha * (labels) * tf.log(tf.sigmoid(model["outputs"]) + 1e-10) + beta * (1.0 - labels) * tf.log(1.0 - tf.sigmoid(model["outputs"]) + 1e-10))
        #loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=labels,predictions=model["outputs"]))
        alpha, beta,accuracy,alpha_op,beta_op=self.Dynamic_Weighted_Binary_CrossEntropy_loss(labels, model["outputs"], alpha, beta)



        grad_update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    
    
        #tensorboard
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("alpha", alpha)
        tf.summary.scalar("beta", beta)
        
        merge_summary = tf.summary.merge_all()
    

        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            session.run(init)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            #saver.restore(session, "./save/char_seg.ckpt-800")
            

            #tensorboard
            summary_writer = tf.summary.FileWriter("./summary/", session.graph)

            epoch=0
            data_generator=get_batch(num_workers=16,image_width=self.input_shape[0],image_height=self.input_shape[1],image_channel=self.input_shape[2], batch_size=self.batch_size)
            
            while True:
                data = next(data_generator)
                x_batch = data[0]
                y_batch = data[1]
                
                feed = {model["inputs"]: x_batch,labels: y_batch}
                #print(x_batch.shape,y_batch.shape)
                learning_rate_train,loss_train,alpha_train, beta_train,accuracy_train,step,summary,_,_,_=session.run([learning_rate,loss,alpha, beta,accuracy,global_step,merge_summary,grad_update,alpha_op,beta_op], feed_dict=feed)
                print("learning rate:%f epoch:%d iter:%d loss:%f alpha_train:%f beta_train:%f accuracy:%f"%(learning_rate_train,epoch,step,loss_train,alpha_train, beta_train,accuracy_train))
                
                #tensorboard
                summary_writer.add_summary(summary, step)

                if step > 0 and step % self.save_steps == 0:
                    save_path = saver.save(session, "save/char_seg.ckpt", global_step=step)
                    print(save_path)
                if step > 0:
                    epoch=step*self.batch_size//self.one_epoch_num

if __name__=="__main__":
    char_seg=Train_char_seg()
    char_seg.train()
