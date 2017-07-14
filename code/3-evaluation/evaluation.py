from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sys
import tables
import os
import numpy as np
from tensorflow.python.ops import control_flow_ops
import random
from nets import nets_factory
from auxiliary import losses
from sklearn.metrics.pairwise import cosine_similarity

slim = tf.contrib.slim

######################
# Train Directory #
######################
tf.app.flags.DEFINE_string(
    'checkpoint_dir', '../../results/TRAIN_CNN_3D',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'evaluation_dataset_path', '../../data/enrollment-evaluation_sample_dataset.hdf5',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'development_dataset_path', '../../data/development_sample_dataset_speaker.hdf5',
    'Directory where checkpoints and event logs are written to.')


tf.app.flags.DEFINE_string(
    'enrollment_dir', '../../results/Model',
    'Directory where checkpoints and event logs are written to.')


tf.app.flags.DEFINE_string(
    'evaluation_dir', '../../results/ROC',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_boolean('online_pair_selection', False,
                            'Use online pair selection.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 20,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 10,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 500,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 1.0, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_string(
    'model_speech', 'cnn_speech', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_epochs', 50, 'The number of epochs for training.')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

# Load the artificial datasets.
fileh = tables.open_file(FLAGS.evaluation_dataset_path, mode='r')
fileh_development = tables.open_file(FLAGS.development_dataset_path, mode='r')
# Train
print("Enrollment data shape:", fileh.root.utterance_enrollment.shape)
print("Enrollment label shape:", fileh.root.label_enrollment.shape)

# Test
print("Evaluation data shape:", fileh.root.utterance_evaluation.shape)
print("Evaluation label shape:",fileh.root.label_evaluation.shape)

# Get the number of subjects
num_subjects_development = len(np.unique(fileh_development.root.label_train[:]))


def main(_):
    # if not FLAGS.dataset_dir:
    #     raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):

        # required from data
        num_samples_per_epoch = fileh.root.label_enrollment.shape[0]
        num_batches_per_epoch = int(num_samples_per_epoch / FLAGS.batch_size)

        num_samples_per_epoch_test = fileh.root.label_evaluation.shape[0]
        num_batches_per_epoch_test = int(num_samples_per_epoch_test / FLAGS.batch_size)

        # Create global_step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        ######################
        # Select the network #
        ######################

        is_training = tf.placeholder(tf.bool)

        model_speech_fn = nets_factory.get_network_fn(
            FLAGS.model_speech,
            num_classes=num_subjects_development,
            is_training=is_training)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        # with tf.device(deploy_config.inputs_device()):
        """
        Define the place holders and creating the batch tensor.
        """
        speech = tf.placeholder(tf.float32, (20, 80, 40, 1))
        label = tf.placeholder(tf.int32, (1))
        batch_dynamic = tf.placeholder(tf.int32, ())
        margin_imp_tensor = tf.placeholder(tf.float32, ())

        # Create the batch tensors
        batch_speech, batch_labels = tf.train.batch(
            [speech, label],
            batch_size=batch_dynamic,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        #############################
        # Specify the loss function #
        #############################
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_clones):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        """
                        Two distance metric are defined:
                           1 - distance_weighted: which is a weighted average of the distance between two structures.
                           2 - distance_l2: which is the regular l2-norm of the two networks outputs.
                        Place holders

                        """
                        ########################################
                        ######## Outputs of two networks #######
                        ########################################
                        # step = int(FLAGS.batch_size / float(FLAGS.num_clones))
                        # logits, end_points_speech = model_speech_fn(batch_speech[i * step : (i + 1) * step])
                        features, logits, end_points_speech = model_speech_fn(batch_speech)

                        # # Uncomment if the output embedding is desired to be as |f(x)| = 1
                        # logits_speech = tf.nn.l2_normalize(logits_speech, dim=1, epsilon=1e-12, name=None)
                        # logits_mouth = tf.nn.l2_normalize(logits_mouth, dim=1, epsilon=1e-12, name=None)

                        #######################################################
                        ################# Distance Calculation ################
                        #######################################################

                        # ##### Weighted distance using a fully connected layer #####
                        # distance_vector = tf.abs(tf.subtract(logits_speech_L, logits_speech_R, name=None))
                        # logits = slim.fully_connected(distance_vector, 2, normalizer_fn=None, activation_fn=None,
                        #                               scope='fc_weighted')

                        ###############################################
                        ########## Loss function ##########
                        ###############################################

                        # one_hot labeling
                        label_onehot = tf.one_hot(tf.squeeze(batch_labels, [1]), depth=num_subjects_development, axis=-1)

                        # Define loss
                        with tf.name_scope('loss'):
                            loss = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_onehot))

                        # Accuracy
                        with tf.name_scope('accuracy'):
                            # Evaluate the model
                            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_onehot, 1))

                            # Accuracy calculation
                            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                            # # ##### call the optimizer ######
                            # # # TODO: call optimizer object outside of this gpu environment
                            # #
                            # # Reuse variables for the next tower.
                            # tf.get_variable_scope().reuse_variables()

        #################################################
        ########### Summary Section #####################
        #################################################

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for all end_points.
        for end_point in end_points_speech:
            x = end_points_speech[end_point]
            summaries.add(tf.summary.scalar('sparsity_speech/' + end_point,
                                            tf.nn.zero_fraction(x)))

            # for end_point in end_points_speech_R:
            #     x = end_points_speech_R[end_point]
            #     summaries.add(tf.summary.scalar('sparsity_mouth/' + end_point,
            #                                     tf.nn.zero_fraction(x)))

            # # Add summaries for variables.
            # for variable in slim.get_model_variables():
            #     summaries.add(tf.summary.histogram(variable.op.name, variable))
            #
            # # # Add to parameters to summaries
            # # summaries.add(tf.summary.scalar('learning_rate', learning_rate))
            # # summaries.add(tf.summary.scalar('global_step', global_step))
            # # summaries.add(tf.summary.scalar('eval/Loss', loss))
            # # summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
            #
            # # Merge all summaries together.
            # summary_op = tf.summary.merge(list(summaries), name='summary_op')

    ###########################
    ######## ######## #########
    ###########################

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # Initialization of the network.
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore, max_to_keep=20)
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ################################################
        ############## ENROLLMENT Model ################
        ################################################

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=FLAGS.checkpoint_dir)
        saver.restore(sess, latest_checkpoint)

        speaker_model_path = os.path.join(FLAGS.enrollment_dir,'MODEL.npy')
        MODEL = np.load(speaker_model_path)



        feature_vector = np.zeros((num_batches_per_epoch_test*FLAGS.batch_size, 128))
        label_vector = np.zeros((num_batches_per_epoch_test * FLAGS.batch_size, 1))

        step = 1
        # Loop over all batches
        for batch_num in range(num_batches_per_epoch_test):

            step += 1
            start_idx = batch_num * FLAGS.batch_size
            end_idx = (batch_num + 1) * FLAGS.batch_size
            speech_evaluation, label_evaluation = fileh.root.utterance_evaluation[start_idx:end_idx, :, :,
                                                  0:1], fileh.root.label_evaluation[start_idx:end_idx]

            # Copy to match dimension
            speech_evaluation = np.repeat(speech_evaluation,20,axis=3)
            speech_evaluation = np.transpose(speech_evaluation[None, :, :, :, :], axes=(1, 4, 2, 3, 0))


            feature = sess.run(
                [features],
                feed_dict={is_training: True, batch_dynamic: label_evaluation.shape[0],
                           batch_speech: speech_evaluation,
                           batch_labels: label_evaluation.reshape([label_evaluation.shape[0], 1])})

            # Extracting the associated numpy array.
            feature_speaker = feature[0]

            # # # L2-norm along each utterance vector
            # feature_speaker = sklearn.preprocessing.normalize(feature_speaker, norm='l2', axis=1, copy=True,
            #                                                   return_norm=False)

            feature_vector[start_idx:end_idx,:] = feature_speaker
            label_vector[start_idx:end_idx,:] = label_evaluation.reshape([label_evaluation.shape[0], 1])


        ########################################
        ########## SCORE COMPUTATION ###########
        ########################################
        NumClasses = MODEL.shape[0]
        NumFeatures = MODEL.shape[1]
        score_vector = np.zeros((feature_vector.shape[0]*NumClasses, 1))
        target_label_vector = np.zeros((feature_vector.shape[0]*NumClasses, 1))


        for i in range(feature_vector.shape[0]):
            if i % 100 ==0:
                print("processing file %d from %d" %(i,feature_vector.shape[0]))
            for j in range(NumClasses):
                model = MODEL[j:j+1, :]
                score = cosine_similarity(feature_vector[i:i+1,:], model)
                score_vector[i*NumClasses + j] = score
                # print(score)
                if (j+1) == label_vector[i,:]:
                    target_label_vector[i*NumClasses + j] = 1
                else:
                    target_label_vector[i * NumClasses + j] = 0

        # Save the score and label vector.
        if not os.path.exists(FLAGS.evaluation_dir):
            os.makedirs(FLAGS.evaluation_dir)
        np.save(os.path.join(FLAGS.evaluation_dir,'score_vector.npy'),score_vector)
        np.save(os.path.join(FLAGS.evaluation_dir,'target_label_vector.npy'),target_label_vector)



if __name__ == '__main__':
    tf.app.run()
