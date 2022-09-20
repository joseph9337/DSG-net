from model import DSGnet
from tool import *
import h5py

def train(args):
    save_path = "./save_model/"
    ## Load Data ##
    num_images = 16000
    f_meta = h5py.File('DATA_DSG.mat')
    mt = f_meta['data']

    RF_v1 = mt['RF1']
    RF_v1 = np.transpose(RF_v1)

    RF_v2 = mt['RF2']
    RF_v2 = np.transpose(RF_v2)

    RF_v3 = mt['RF3']
    RF_v3 = np.transpose(RF_v3)

    RF_v4 = mt['RF4']
    RF_v4 = np.transpose(RF_v4)

    label_mt = mt['atten']
    label_mt = np.transpose(label_mt)
    label_mt = np.array(label_mt, np.float32) / 255

    BM_v1 = mt['b_mode1']
    BM_v1 = np.transpose(BM_v1)
    BM_v1 = np.array(BM_v1, np.float32) / 255

    BM_v2 = mt['b_mode2']
    BM_v2 = np.transpose(BM_v2)
    BM_v2 = np.array(BM_v2, np.float32) / 255

    BM_v3 = mt['b_mode3']
    BM_v3 = np.transpose(BM_v3)
    BM_v3 = np.array(BM_v3, np.float32) / 255

    BM_v4 = mt['b_mode4']
    BM_v4 = np.transpose(BM_v4)
    BM_v4 = np.array(BM_v4, np.float32) / 255


    batch_size = args.batch_size
    total_steps = int(num_images)+1

    model = DSGnet(args.input_shape, args.label_shape, args.DSA_position,args.Model_version,args.batch_size,args.learning_rate1,args.learning_rate2,drop_out=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("model initialized")

    saver = tf.train.Saver(max_to_keep=None)
    # saver.restore(sess, "save_model/model_#.cptk")

    for epoch in range(0,args.epoch):
        if args.drop_out == "True": model.drop_out = True
        loss_sum = 0
        b = 0
        for i in range(total_steps):

            [label_batch, data1_SRCA,data2_SRCA,data3_SRCA,data4_SRCA,data5_SRCA,data6_SRCA,data7_SRCA, bmode_SRCA,\
           data1_SRCB,data2_SRCB,data3_SRCB,data4_SRCB,data5_SRCB,data6_SRCB,data7_SRCB, bmode_SRCB,\
           data1_META,data2_META,data3_META,data4_META,data5_META,data6_META,data7_META, bmode_META] = Data_load(RF_v1, RF_v2, RF_v3,RF_v4, BM_v1,BM_v2,BM_v3,BM_v4,label_mt)

            feed_dict = {model.input1_SRCA: np.float32(data1_SRCA),model.input2_SRCA: np.float32(data2_SRCA),model.input3_SRCA: np.float32(data3_SRCA),model.input4_SRCA: np.float32(data4_SRCA),
                         model.input5_SRCA: np.float32(data5_SRCA),model.input6_SRCA: np.float32(data6_SRCA),model.input7_SRCA: np.float32(data7_SRCA),model.BM_SRCA: np.float32(bmode_SRCA),
                         model.input1_SRCB: np.float32(data1_SRCB), model.input2_SRCB: np.float32(data2_SRCB),
                         model.input3_SRCB: np.float32(data3_SRCB), model.input4_SRCB: np.float32(data4_SRCB),
                         model.input5_SRCB: np.float32(data5_SRCB), model.input6_SRCB: np.float32(data6_SRCB),
                         model.input7_SRCB: np.float32(data7_SRCB), model.BM_SRCB: np.float32(bmode_SRCB),
                         model.input1_META: np.float32(data1_META), model.input2_META: np.float32(data2_META),
                         model.input3_META: np.float32(data3_META), model.input4_META: np.float32(data4_META),
                         model.input5_META: np.float32(data5_META), model.input6_META: np.float32(data6_META),
                         model.input7_META: np.float32(data7_META), model.BM_META: np.float32(bmode_META), model.labels: np.float32(label_batch)}



            train_sample, loss,loss_DSA,loss_TV,_, _ = sess.run([model.logits,  model.loss, model.RF_loss,model.loss_TV, model.training, model.meta_training], feed_dict = feed_dict)

            loss_sum += loss
            b+=batch_size

        if (epoch + 1) % 10 == 0:
            print("Saving model...")
            saver.save(sess, save_path + "model_" + str(epoch + 1) + ".cptk")
        if (epoch+1) % 1 == 0:
            reshaped_label = np.reshape(label_batch, (128, 128))
            reshaped_sample = np.reshape(train_sample, (128, 128))
            cv2.imwrite(
                "./samples/train_label_%d.png" % (epoch),reshaped_label)
            cv2.imwrite(
                "./samples/train_sample_%d.png" % (epoch),reshaped_sample)


def test(args):
    model = DSGnet(args.input_shape, args.label_shape,args.DSA_position,args.Model_version,args.batch_size,args.learning_rate1, args.learning_rate2, drop_out=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("model initialized")
    # f_meta = h5py.File('DATA_TEST.mat')
    # mt = f_meta['data']
    #
    # RF = mt['RF']
    # RF = np.transpose(RF)
    #
    # BM = mt['b_mode']
    # BM = np.transpose(BM)
    #
    # saver = tf.train.Saver(max_to_keep=None)
    # saver.restore(sess, "save_model/model_#.cptk")
    # icd = 0
    # cond = np.zeros((1, 4))
    # cond[0, icd] = 1
    # feed_dict = {model.input1_SRCA: np.float32(RF[:,:,:,0]), model.input2_SRCA: np.float32(RF[:,:,:,1]),
    #              model.input3_SRCA: np.float32(RF[:,:,:,2]), model.input4_SRCA: np.float32(RF[:,:,:,3]),
    #              model.input5_SRCA: np.float32(RF[:,:,:,4]), model.input6_SRCA: np.float32(RF[:,:,:,5]),
    #              model.input7_SRCA: np.float32(RF[:,:,:,6]), model.BM_SRCA: np.float32(BM),
    #              model.cond_ph: np.float32(cond)}
    #
    # output= sess.run(
    #     [model.logits], feed_dict=feed_dict)

