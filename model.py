import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils import pytorch_xavier_weight_factor, pytorch_kaiming_weight_factor, SpatialTransformer
factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
weight_regularizer = None
from toimg import toimg
class DSGnet():
    def __init__(self, input_shape, output_shape,DSA_position, Model_version,batch_size, lr1, lr2, drop_out=False):
        if input_shape != output_shape: self.padding = "same"
        else: self.padding = "same"
        self.spatial_transformer = SpatialTransformer(name='transformer')
        self.DSA_position = DSA_position
        self.Model_version = Model_version
        self.input1_SRCA = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input2_SRCA = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input3_SRCA = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input4_SRCA = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input5_SRCA = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input6_SRCA = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input7_SRCA = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.BM_SRCA = tf.placeholder(tf.float32, [batch_size] + input_shape)

        self.input1_SRCB = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input2_SRCB = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input3_SRCB = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input4_SRCB = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input5_SRCB = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input6_SRCB = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input7_SRCB = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.BM_SRCB = tf.placeholder(tf.float32, [batch_size] + input_shape)

        self.input1_META = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input2_META = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input3_META = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input4_META = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input5_META = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input6_META = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.input7_META = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.BM_META = tf.placeholder(tf.float32, [batch_size] + input_shape)
        self.labels = tf.placeholder(tf.float32, [batch_size] + output_shape)

        self.weights = {}

        self.drop_out = drop_out
        self.Weight_Init()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr1)

        self.logits, self.enc_SRCA, self.disp_SRCA, self.logits_16, self.logits_32, self.logits_64, self.logits_128 = self.Inference(self.input1_SRCA, self.input2_SRCA, self.input3_SRCA, self.input4_SRCA,
                                                              self.input5_SRCA, self.input6_SRCA, self.input7_SRCA, self.BM_SRCA,
                                                              self.weights)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.logits))
        self.training = self.optimizer.minimize(self.loss)
        self.low_loss = tf.reduce_mean(tf.square(tf.image.resize_nearest_neighbor(self.labels, (16, 16)) - self.logits_16)) \
                        + tf.reduce_mean(tf.square(tf.image.resize_nearest_neighbor(self.labels, (32, 32)) - self.logits_32)) \
                        + tf.reduce_mean(tf.square(tf.image.resize_nearest_neighbor(self.labels, (64, 64)) - self.logits_64)) \
                        + tf.reduce_mean(tf.square(tf.image.resize_nearest_neighbor(self.labels, (128, 128)) - self.logits_128))

        self.logits_tgt, self.enc_SRCB, self.disp_SRCB, _, _, _, _ = self.Inference(self.input1_SRCB, self.input2_SRCB, self.input3_SRCB,
                                                                  self.input4_SRCB,
                                                                  self.input5_SRCB, self.input6_SRCB, self.input7_SRCB,
                                                                  self.BM_SRCB, self.weights)

        self.loss_TV = self.total_variation_loss(self.disp_SRCA) + self.total_variation_loss(self.disp_SRCB)
        self.loss_DSA = self.RF_loss(self.enc_SRCA, self.enc_SRCB)
        self.loss_reg = self.regularization_loss()
        self.MLSD(self.loss_DSA, lr2)
        self.loss_training = self.loss + self.low_loss + self.loss_reg*1e-6
        self.training = self.optimizer.minimize(self.loss_training)

        self.meta_training = self.optimizer.minimize(self.loss_meta)

    def MLSD(self, loss_enc, lr2):
        grads = tf.gradients(loss_enc, list(self.weights.values()))
        gvs = dict(zip(self.weights.keys(), grads))
        tmp_list = []
        for key in gvs.keys():
            if not gvs[key] == None:
                tmp_list.append(self.weights[key] - lr2 * gvs[key])
            else:
                tmp_list.append(self.weights[key])
        fast_weights = dict(zip(self.weights.keys(),
                                tmp_list))
        self.id = 0
        self.id_DSA = 0
        self.logits_meta, self.enc_META, self.disp_META, _, _, _, _ = self.Inference(self.input1_META, self.input2_META, self.input3_META, self.input4_META,
                                                                     self.input5_META, self.input6_META, self.input7_META,self.BM_META,fast_weights)
        self.loss_meta = self.RF_loss(self.enc_SRCA,self.enc_META) + loss_enc
        return self.loss_meta

    def Weight_Init(self):
        self.initalize = True
        _, _, _, _, _, _, _= self.Inference(self.input1_SRCA, self.input2_SRCA,
                                                                    self.input3_SRCA, self.input4_SRCA,
                                                                    self.input5_SRCA, self.input6_SRCA,
                                                                    self.input7_SRCA, self.BM_SRCA,
                                                                    self.weights)
        self.initalize = False

    def Inference(self,input1, input2, input3, input4, input5, input6, input7,BM, weights = None):

        self.id = 0
        self.id_DSA = 0

        with tf.variable_scope('DSA', reuse=tf.AUTO_REUSE):
            channel=16
            xA = input1
            xA = self.conv_encode12(xA, channel, weights=weights,  pad=1, use_bias=True, sn=True, scope='conv')
            xB = input2
            xB = self.conv_encode12(xB, channel, weights=weights, pad=1, use_bias=True, sn=True, scope='convB')
            x=tf.concat([xA, xB], axis=-1)
            xC = input3
            xC = self.conv_encode12(xC, channel, weights=weights,   pad=1, use_bias=True, sn=True, scope='convC')
            x=tf.concat([x, xC], axis=-1)
            xD = input4
            xD = self.conv_encode12(xD, channel, weights=weights,   pad=1, use_bias=True, sn=True, scope='convD')
            x=tf.concat([x, xD], axis=-1)
            xE = input5
            xE = self.conv_encode12(xE, channel, weights=weights,  pad=1, use_bias=True, sn=True, scope='convE')
            x=tf.concat([x, xE], axis=-1)
            xF = input6
            xF = self.conv_encode12(xF, channel, weights=weights,  pad=1, use_bias=True, sn=True, scope='convF')
            x=tf.concat([x, xF], axis=-1)
            xG = input7
            xG = self.conv_encode12(xG, channel, weights=weights,  pad=1, use_bias=True, sn=True, scope='convG')
            x=tf.concat([x, xG], axis=-1)
            if self.DSA_position == 'first':
                x,disp = self.Spatial_trainsform_net(x,BM, channel,weights)
                tmp=x

        with tf.variable_scope('Baseline', reuse=tf.AUTO_REUSE):

            for i in range(2):
                x = tf.keras.activations.relu(x)
                x = self.conv_encode12(x, channel * 2, weights=weights,   pad=1, use_bias=True, sn=True, scope='conv_' + str(i))
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(rate=0.5)(x)
                channel = channel * 2

            if self.DSA_position == 'middle':
                x, disp = self.Spatial_trainsform_net(x, BM, channel, weights)
                tmp = x

            for i in range(3):
                x = tf.keras.activations.relu(x)
                x = self.conv_encode(x, channel * 2, weights=weights,   pad=1, stride =2, use_bias=True, sn=True, scope='conv_2' + str(i))
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(rate=0.5)(x)
                print(x)
                channel = channel*2

            if self.DSA_position == 'end':
                x, disp = self.Spatial_trainsform_net(x, BM, channel, weights)
                tmp = x

            x = tf.keras.activations.relu(x)
            x = tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]])
            ext =  tf.layers.conv2d(inputs=x, filters=channel,
                                         kernel_size=3, kernel_initializer=weight_init,
                                         kernel_regularizer=weight_regularizer,
                                         strides=[2,3])
            ext = tf.keras.layers.Dropout(rate=0.5)(ext)
            ext = tf.keras.activations.relu(ext)

            x,I16,I32,I64,I128 = self.Decoder_module(ext, weights=weights)
        return x, tmp, disp,I16,I32,I64,I128


    def resblk(self,input,channel = 64, weights=None ):
        x = self.conv(input, channel, weights=weights)
        x = self.conv_noact(x, channel, weights=weights)
        x = x+input
        x = tf.keras.activations.relu(x)
        return x

    def sinblk(self, input, channel = 64, weights=None):
        if self.Model_version=="D1":
            block_num=1
        elif self.Model_version=="D2":
            block_num=2
        elif self.Model_version == "D3":
            block_num=3
        elif self.Model_version == "D4":
            block_num=4
        x = self.resblk(input, channel, weights=weights)
        for i in range(block_num):
            x = self.resblk(x, channel, weights=weights)
        return x



    def Decoder_module(self, input, scope=None, weights=None):
        with tf.variable_scope('scope'):
            x = self.conv(input, 512, weights=weights)
            ## first
            x_16_1 = self.sinblk(x, 512, weights=weights)

            ## second
            x_16_2 = x_16_1
            x_32_2 = self.up(x_16_1, 256, weights=weights)
            x_16_2 = self.sinblk(x_16_2, 512, weights=weights)
            x_32_2 = self.sinblk(x_32_2, 256, weights=weights)

            ## third
            x_16_3 = tf.concat([x_16_2, self.dn(x_32_2, 64, weights=weights)], -1)
            x_16_3 = self.conv(x_16_3, 512, weights=weights)

            x_32_3 = tf.concat([x_32_2, self.up(x_16_2, 128, weights=weights)], -1)
            x_32_3 = self.conv(x_32_3, 256, weights=weights)

            x_64_3 = tf.concat([self.up(x_32_3, 128, weights=weights), self.up(self.up(x_16_2, 256, weights=weights), 128, weights=weights)], -1)
            x_64_3 = self.conv(x_64_3, 128, weights=weights)

            x_16_3 = self.sinblk(x_16_3, 512, weights=weights)
            x_32_3 = self.sinblk(x_32_3, 256, weights=weights)
            x_64_3 = self.sinblk(x_64_3, 128, weights=weights)


            # fourth

            x_16_4 = tf.concat([self.dn(x_32_3, 512, weights=weights), self.dn(self.dn(x_64_3, 256, weights=weights), 512, weights=weights)], -1)
            x_16_4 = tf.concat([x_16_3, x_16_4], -1)
            x_16_4 = self.conv(x_16_4, 512, weights=weights)

            x_32_4 = tf.concat([self.up(x_16_3, 256, weights=weights), self.dn(x_64_3, 256, weights=weights)], -1)
            x_32_4 = tf.concat([x_32_4, x_32_3], -1)
            x_32_4 = self.conv(x_32_4, 256, weights=weights)

            x_64_4 = tf.concat([self.up(x_32_3, 128, weights=weights), self.up(self.up(x_16_3, 256, weights=weights), 128, weights=weights)], -1)
            x_64_4 = tf.concat([x_64_4, x_64_3], -1)
            x_64_4 = self.conv(x_64_4, 128, weights=weights)

            x_128_4 = tf.concat([self.up(self.up(x_32_3, 128, weights=weights), 64, weights=weights), self.up(self.up(self.up(x_16_3, 256, weights=weights), 128, weights=weights), 64, weights=weights)], -1)
            x_128_4 = tf.concat([x_128_4, self.up(x_64_3, 64, weights=weights)], -1)
            x_128_4 = self.conv(x_128_4, 64, weights=weights)

            x_128_4 = self.sinblk(x_128_4, 64, weights=weights)
            x_64_4 = self.sinblk(x_64_4, 128, weights=weights)
            x_32_4 = self.sinblk(x_32_4, 256, weights=weights)
            x_16_4 = self.sinblk(x_16_4, 512, weights=weights)

            with tf.variable_scope('toimg', reuse=tf.AUTO_REUSE):
                I16=toimg(x_16_4,16)
                I32=toimg(x_32_4,32)
                I64=toimg(x_64_4,64)
                I128=toimg(x_128_4,128)
            ## Final

            x_128_5 = tf.concat([self.up(self.up(x_32_4, 128, weights=weights), 64, weights=weights), self.up(self.up(self.up(x_16_4, 256, weights=weights), 128, weights=weights), 64, weights=weights)], -1)
            x_128_5 = tf.concat([x_128_5, self.up(x_64_4, 64, weights=weights)], -1)
            x_128_5 = tf.concat([x_128_5, x_128_4], -1)
            x_128_5 = self.conv(x_128_5, 64, weights=weights)
            x_128_5 = self.sinblk(x_128_5, 64, weights=weights)
            x = self.conv(x_128_5, 1, weights=weights, kernel_size=1, activation=tf.keras.activations.sigmoid)
        return x, I16, I32, I64, I128

    def Spatial_trainsform_net(self, input, BM, channel, weights):

        sx = int(input.get_shape().as_list()[1])
        sy = int(input.get_shape().as_list()[2])
        sc = int(input.get_shape().as_list()[3])
        BM = tf.image.resize_nearest_neighbor(BM, (sx, sy))
        RF = input
        x = self.conv_DSA(input, channel, weights=weights)
        x = self.conv_DSA(x, channel, weights=weights)

        BM = self.conv_DSA(BM, channel, weights=weights)
        BM = self.conv_DSA(BM, channel, weights=weights)
        x = tf.concat([x, BM], axis=-1)

        x = self.conv_DSA(x, channel, weights=weights)
        x = self.conv_DSA(x, channel, weights=weights)

        x = self.conv_DSA(x, channel, weights=weights)
        x = self.conv_DSA(x, channel, weights=weights)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
        x = self.conv_DSA(x, channel * 2, weights=weights)
        x = self.conv_DSA(x, channel * 2, weights=weights)
        x_2 = x
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
        x = self.conv_DSA(x, channel * 4, weights=weights)
        x = self.conv_DSA(x, channel * 4, weights=weights)
        x_4 = x
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
        x = self.conv_DSA(x, channel * 4, weights=weights)
        x = self.conv_DSA(x, channel * 4, weights=weights)
        x = self.up_DSA(x, channel * 4, weights=weights)
        # x = tf.keras.layers.ZeroPadding2D(
        #     padding=((0, 0), (0, 1))
        # )(x)
        x = tf.concat([x, x_4], axis=-1)
        x = self.conv_DSA(x, channel * 4, weights=weights)
        x = self.conv_DSA(x, channel * 4, weights=weights)
        x = self.up_DSA(x, channel * 4, weights=weights)

        x = tf.keras.layers.ZeroPadding2D(
            padding=((0, 0), (0, 1))
        )(x)
        x = tf.concat([x, x_2], axis=-1)
        x = self.conv_DSA(x, channel * 4, weights=weights)
        x = self.conv_DSA(x, channel * 4, weights=weights)
        x = self.up_DSA(x, channel * 4, weights=weights)
        x = self.conv_DSA(x, channel * 4, weights=weights)
        x = self.conv_DSA(x, channel * 4, weights=weights)
        x = self.conv_DSA_noact(x, 2, weights=weights)
        # x = tf.keras.layers.ZeroPadding2D(
        #     padding=((0, 0), (0, 1))
        # )(x)
        mask = tf.concat([tf.ones([1, sx, sy, 1]), tf.zeros([1, sx, sy, 1])], axis=-1)
        x = x * mask
        disp = x
        x = self.spatial_transformer([RF, x])
        x = tf.reshape(x, [1, sx, sy, sc])
        return x, disp



    def up(self, input, channel, weights=None):
        x = tf.keras.layers.Conv2DTranspose(filters = channel, kernel_size = 3, strides = [2, 2], padding = self.padding, activation = None)(input)
        x = tf.keras.layers.Conv2D(filters=channel, kernel_size=1, strides=[1, 1],
                                    activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        return x


    def dn(self, input, channel, weights=None):
        if self.initalize == True:
            self.weights[str(self.id)] = tf.get_variable("kernel"+str(self.id), shape=[3, 3, input.get_shape()[-1], channel],
                                                         regularizer=weight_regularizer)
        x = tf.nn.conv2d(input=input, filter=self.weights[str(self.id)],
                         strides=[1, 2, 2, 1], padding='SAME')
        if self.initalize == True:
            self.weights['b' + str(self.id)] = tf.get_variable("bias"+str(self.id), [channel],
                                                           initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, self.weights['b' + str(self.id)])

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.activations.relu(x)


        self.id = self.id + 1
        return x


    def conv(self, input, num_outputs, weights=None, kernel_size=3, stride=1, activation=tf.keras.activations.relu):
        if self.initalize == True:
            self.weights[str(self.id)] = tf.get_variable("kernel"+str(self.id), shape=[3, 3, input.get_shape()[-1], num_outputs],trainable = True,
                                                         regularizer=weight_regularizer)
        x = tf.nn.conv2d(input=input, filter=self.weights[str(self.id)],
                         strides=[1, 1, 1, 1], padding='SAME')
        if self.initalize == True:
            self.weights['b' + str(self.id)] = tf.get_variable("bias"+str(self.id), [num_outputs],trainable = True,
                                                               initializer=tf.constant_initializer(0.0))
        tmp = tf.nn.bias_add(x, self.weights['b' + str(self.id)])
        tmp = tf.keras.layers.BatchNormalization()(tmp)
        tmp = activation(tmp)
        self.id = self.id + 1

        if self.drop_out == True: return tf.keras.layers.Dropout(rate=0.5)(tmp)
        else: return tmp

    def conv_DSA(self, input, num_outputs, weights=None, kernel_size=3, stride=1, activation=tf.keras.activations.relu):
        with tf.variable_scope('DSA', reuse=tf.AUTO_REUSE):
            if self.initalize == True:
                self.weights[str(self.id_DSA)+"DSA"] = tf.get_variable("kernel"+"DSA" + str(self.id_DSA),
                                                             shape=[3, 3, input.get_shape()[-1], num_outputs],
                                                             regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=input, filter=self.weights[str(self.id_DSA)+"DSA"],
                             strides=[1, 1, 1, 1], padding='SAME')
            if self.initalize == True:
                self.weights['b' + str(self.id_DSA)+"DSA"] = tf.get_variable("bias"+"DSA" + str(self.id_DSA), [num_outputs], trainable=True,
                                                                   initializer=tf.constant_initializer(0.0))
            tmp = tf.nn.bias_add(x, self.weights['b' + str(self.id_DSA)+"DSA"])
            tmp = tf.keras.layers.BatchNormalization()(tmp)
            tmp = activation(tmp)
        self.id_DSA = self.id_DSA + 1

        if self.drop_out == True: return tf.keras.layers.Dropout(rate=0.5)(tmp)

        else: return tmp

    def conv_DSA_noact(self, input, num_outputs, weights=None, kernel_size=3, stride=1):
        with tf.variable_scope('DSA', reuse=tf.AUTO_REUSE):

            self.weights[str(self.id_DSA)+"DSA"] = tf.get_variable("kernel"+"DSA" + str(self.id_DSA),
                                                         shape=[3, 3, input.get_shape()[-1], num_outputs],
                                                         regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=input, filter=self.weights[str(self.id_DSA)+"DSA"],
                             strides=[1, 1, 1, 1], padding='SAME')
            self.weights['b' + str(self.id_DSA)+"DSA"] = tf.get_variable("bias"+"DSA" + str(self.id_DSA), [num_outputs], trainable=True,
                                                               initializer=tf.constant_initializer(0.0))
            tmp = tf.nn.bias_add(x, self.weights['b' + str(self.id_DSA)+"DSA"])
            tmp = tf.keras.layers.BatchNormalization()(tmp)
        self.id_DSA = self.id_DSA + 1
        if self.drop_out == True:
            return tf.keras.layers.Dropout(rate=0.5)(tmp)
        else:
            return tmp




    def conv_noact(self, input, num_outputs, weights=None, kernel_size=3, stride=1):
        if self.initalize == True:
            self.weights[str(self.id)] = tf.get_variable("kernel"+str(self.id), shape=[3, 3, input.get_shape()[-1], num_outputs],
                                                         regularizer=weight_regularizer)
        x = tf.nn.conv2d(input=input, filter=self.weights[str(self.id)],
                         strides=[1, 1, 1, 1], padding='SAME')
        if self.initalize == True:
            self.weights['b' + str(self.id)] = tf.get_variable("bias"+str(self.id), [num_outputs],trainable = True,
                                                               initializer=tf.constant_initializer(0.0))
        tmp = tf.nn.bias_add(x, self.weights['b' + str(self.id)])
        tmp = tf.keras.layers.BatchNormalization()(tmp)
        self.id = self.id + 1
        if self.drop_out == True: return tf.keras.layers.Dropout(rate=0.5)(tmp)
        else: return tmp


    def up_DSA(self, input, num_outputs, weights=None, kernel_size=3, stride=1, activation=tf.keras.activations.relu):
        with tf.variable_scope('DSA', reuse=tf.AUTO_REUSE):
            if self.initalize == True:
                self.weights[str(self.id_DSA)+"DSA"] = tf.get_variable("kernel"+"DSA" + str(self.id_DSA),
                                                             shape=[3, 3, input.get_shape()[-1], num_outputs],
                                                             regularizer=weight_regularizer)

            x = tf.nn.conv_transpose(
                input,
                self.weights[str(self.id_DSA) + "DSA"],
                [input.get_shape()[0],input.get_shape()[1]*2,input.get_shape()[2]*2,input.get_shape()[3]],
                [1, 2, 2, 1],
                padding='SAME',
                data_format=None,
                dilations=None,
                name=None
            )
            if self.initalize == True:
                self.weights['b' + str(self.id_DSA)+"DSA"] = tf.get_variable("bias"+"DSA" + str(self.id_DSA), [num_outputs], trainable=True,
                                                                   initializer=tf.constant_initializer(0.0))
            tmp = tf.nn.bias_add(x, self.weights['b' + str(self.id_DSA)+"DSA"])
        self.id_DSA = self.id_DSA + 1

        if self.drop_out == True: return tf.keras.layers.Dropout(rate=0.5)(tmp)

        else: return tmp

    def conv_encode(self,x, channels, weights=None,kernel=3, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
        factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
        weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
        # tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        weight_regularizer = None
        weight_regularizer_fully = None
        with tf.variable_scope(scope):
            if pad > 0:
                h = x.get_shape().as_list()[1]
                if h % 2 == 0:
                    pad = pad * 2
                else:
                    pad = max(kernel - (h % stride), 0)

                pad_top = pad // 2
                pad_bottom = pad - pad_top
                pad_left = pad // 2
                pad_right = pad - pad_left

                if pad_type == 'zero':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                if pad_type == 'reflect':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
            if self.initalize == True:
                self.weights[str(self.id)] = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],trainable = True,
                                    initializer=weight_init,
                                    regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=self.weights[str(self.id)],
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                if self.initalize == True:
                    self.weights['b'+str(self.id)]  = tf.get_variable("bias", [channels], trainable = True,initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, self.weights['b'+str(self.id)])
        self.id = self.id+1
        return x

    def conv_encode12(self, x, channels, weights=None,kernel=3, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False,
                       scope='conv_0'):
        factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
        weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
        # tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        weight_regularizer = None
        weight_regularizer_fully = None
        with tf.variable_scope(scope):
            if pad > 0:
                h = x.get_shape().as_list()[1]
                if h % 2 == 0:
                    pad = pad * 2
                else:
                    pad = max(kernel - (h % stride), 0)

                pad_top = pad // 2
                pad_bottom = pad - pad_top
                pad_left = pad // 2
                pad_right = pad - pad_left

                if pad_type == 'zero':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                if pad_type == 'reflect':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
            if self.initalize == True:
                self.weights[str(self.id)] = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],trainable = True,
                                                             initializer=weight_init,
                                                             regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=self.weights[str(self.id)],
                             strides=[1, 1, 2, 1], padding='VALID')
            if use_bias:
                if self.initalize== True:
                    self.weights['b' + str(self.id)] = tf.get_variable("bias", [channels],trainable = True,
                                                                   initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, self.weights['b' + str(self.id)])
        self.id = self.id + 1
        return x



    def regularization_loss(self):
        """
        If you want to use "Regularization"
        g_loss += regularization_loss('generator')
        d_loss += regularization_loss('discriminator')
        """
        collection_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        loss = []
        for item in collection_regularization:
            loss.append(item)

        return tf.reduce_sum(loss)

    def total_variation_loss(self, output):
        tv_loss = tf.reduce_mean(tf.abs(
            (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]))) / 2.0
        return tv_loss

    def RF_loss(self,SRCA, SRCB):
        loss = tf.reduce_mean(
            tf.square(SRCA - SRCB) / tf.reduce_mean(tf.square(SRCA))) + tf.reduce_mean(
            tf.square(SRCA - SRCB) / tf.reduce_mean(tf.square(SRCB)))

        return loss