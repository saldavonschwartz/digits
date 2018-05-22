Digits: A neural net classifier trained in Python and deployed to iOS via Objective-C++
=======================================================================================

This project shows how to train and deploy a classifier for handwritten digits.
For more info on the training procedure see `this article <https://0xfede.io/2018/05/16/digits.html>`_.

The project is divided into 2 parts:

Training and Testing in Python (*python* folder):
-------------------------------------------------
1. Models were generated for a combination of hyper parameters and topologies.
2. These models were trained and validated against the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset.
3. The best 5 models, including intermediates saved through early stopping, were kept and tested.
4. The best 5 models were further tested with custom images taken with the same camera of the target iOS device. For this step OpenCV was used to extract ROIs.

Due to space considerations, the project contains 2 of the best 5 models, along with the stats for the whole training session of all models.

One of these 2 models achieved a test accuracy of 99.18% but predicted one custom image wrong.
The other model achieved a test accuracy of 99% but predicted all custom images right and is significantly smaller than the other model.
Consequently the smaller model was deployed to iOS.


Deploying to iOS (*ios* folder):
--------------------------------
An iOS app was implemented to take pictures and feed them to the trained model.

1. The app takes a picture with the device's camera.
2. The CG image is converted to an OpenCV matrix and ROIs are extracted as in step 4 of the Python part.
3. The resulting image is fed to the trained model for prediction.

For step 3, a forward-only subset of the Python neural net framework (NNKit) was implemented in C++.

Dependencies:
=============

Python:
-------
* `Numpy <http://www.numpy.org>`_
* `NNKit <https://github.com/saldavonschwartz/nnkit>`_
* `OpenCV <https://opencv.org>`_ (if you want to preprocess custom images before testing).
* `matplotlib <www.apple.com>`_ (if you want to plot training stats).

iOS:
----
* `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_ (header only, included in project).
* `JSON for Modern C++ <https://github.com/nlohmann/json>`_ (header only, included in project).
* `OpenCV <https://opencv.org>`_ (iOS framework only).
* `XCode <https://developer.apple.com/xcode/>`_

Installation:
=============
After downloading or cloning the repo:

* Python: just `pip install -r requirements.txt`
* iOS / Xcode: download the OpenCV framework and link against it in the Xcode project.
