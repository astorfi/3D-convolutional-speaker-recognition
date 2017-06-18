=====================================
3D-convolutional-speaker-recognition
=====================================

This repository contains the code release for our paper titled as *"Text-Independent
Speaker Verification Using 3D Convolutional Neural Networks"*. If you would use it, please cite as follows:

.. code:: shell

  @article{torfi2017text,
    title={Text-Independent Speaker Verification Using 3D Convolutional Neural Networks},
    author={Torfi, Amirsina and Nasrabadi, Nasser M and Dawson, Jeremy},
    journal={arXiv preprint arXiv:1705.09422},
    year={2017}
  }

.. _TensorFlow: https://www.tensorflow.org/

The code has been developed using TensorFlow_. The input pipeline must be prepaired by the users.
This code is aimed to provide the implementation for Speaker Verification (SR) by using 3D convolutional neural networks
following the SR protocol.


-------------
General View
-------------

We leveraged 3D convolutional architecture for creating the speaker model in order to simeoultaneously
capturing the speech-realted and temporal information from the speakers' utterances.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Speaker Verification Protocol(SVP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this work, a 3D Convolutional Neural Network (3D-CNN)
architecture has been utilized for text-independent speaker
verification in three phases.

     1. At the **development phase**, a CNN is trained
     to classify speakers at the utterance-level.

     2. In the **enrollment stage**, the trained network is utilized to directly create a
     speaker model for each speaker based on the extracted fea-
     tures.

     3. Finally, in the **evaluation phase**, the extracted features
     from the test utterance will be compared to the stored speaker
     model to verify the claimed identity.

The aformentioned three phases, are usually considered as the SV protocol. One of the main
challenges is the creation of the speaker models. Previously-reported approaches create
speaker models based on averaging the extracted features from utterances of the speaker,
which is known as a d-vector system.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
How to leverage 3D Convolutional Neural Networks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In our paper, we propose to use the 3D-CNNs for direct speaker model creation
in which, for both development and enrollment phases, an identical number of
speaker utterances is fed to the network for representing the speaker utterances
and creation of the speaker model. This leads to simultaneously capturing the
speaker-related information and building a more robust system to cope with
within-speaker variation. We demonstrate that the proposed method significantly
outperforms the d-vector verification system.


--------------------
Code Implementation
--------------------

The input pipline must be provided by the user. The rest of the implementation consider the dataset
which contains the utterance-based extracted features are stored in a ``HDF5`` file. However, this
is not a necessity becasue by following the code, it can be seen that the experiments can be done by
any file format as long as it is adaptible with ``TensorFlow``.

~~~~~~~~~~~~~~~
Input Pipeline
~~~~~~~~~~~~~~~
