Digits: A neural net classifier trained in Python and deployed to iOS via Objective-C++
=======================================================================================

This project shows how to generate and train a series of handwritten digit classifiers in Python and deploy a final model (99% accuracy) in iOS.
For more info on the training procedure and results see `this post <https://0xfede.io/2018/05/16/digits.html>`_.

The project is divided into 2 parts:

Part 1: Training and Testing in Python (*python* folder):
---------------------------------------------------------
1. Models were generated for a combination of hyper parameters and topologies and trained on the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset.
2. The best 5 models were kept and tested both on a portion of the MNIST test set as well as custom images.
3. A final model was chosen out of the 5 to deploy to iOS.

Due to space considerations, the project only contains the final model, along with stats for the whole training session of all models.

**Contents:**

- :code:`digits.py`: the entry point for training and testing.
- :code:`training.py`: the training algorithm, including topology + hyperparameter search.
- :code:`testing.py`: testing algorithms for both MNIST as well as custom images, including OpenCV code to preprocess custom images.
- :code:`statsplot.py`: plotting of training statistics.
- *data*: the MNIST dataset plus custom images used in testing.
- *training*: a copy of the the final model and training stats from the session in which the final model as well as many others were generated.


Part 2: Deploying to iOS (*ios* folder):
-----------------------------------------
An iOS app was implemented to take photographs, preprocess them and run the trained model on them to predict digits.

**Contents:**

- :code:`ViewController`: Manages both the UI of the app and an instance of :code:`NNModel` to do all preprocessing and prediction.
- :code:`NNModel`: The iOS-side model, written in Objective-C++ to interoperate between Cocoa, OpenCV and the trained model.
- :code:`nn.hpp`, :code:`arithmetic.hpp`, :code:`activation.hpp`: C++ implementation of subset of NNKit (originally written in Python), necessary to use the trained model in iOS.
- *3rdparty*: C++ libraries used by the NNKit C++ subset.


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

* Python: just :code:`pip install -r requirements.txt`
* iOS / Xcode: download the OpenCV framework and link against it in the Xcode project.
