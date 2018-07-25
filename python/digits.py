# The MIT License (MIT)
#
# Copyright (c) 2018 Federico Saldarini
# https://www.linkedin.com/in/federicosaldarini
# https://github.com/saldavonschwartz
# https://0xfede.io
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import nnkit as nn
import json, glob, os

import testing
import training
import statsplot


def loadMNISTData(path):
    """Load the MNIST dataset and make a validation set out of the test set.

    :param path: path to the compressed dataset.

    :return: a list of 3 2-tuples where for each tuple, the first element contains examples and the second element
    contains the target labels for each example. Tuples are for the training, validation and test sets.
    """
    sets = []

    with np.load(path) as data:
        sets.append([data["train_images"], data["train_labels"]])
        sets.append([data["test_images"], data["test_labels"]])

    # Make outputs into one-hot vectors suitable for a 10 unit softmax layer:
    for s in sets:
        size = s[0].shape[0], 10
        hot = np.zeros(size)
        hot[range(size[0]), s[1]] = 1
        s[1] = hot

    # Use half the test set for validation:
    testSet = sets[1]
    halfSize = len(testSet[0]) // 2
    validationSet = [testSet[0][0:halfSize], testSet[1][0:halfSize]]
    testSet = [testSet[0][halfSize:], testSet[1][halfSize:]]

    return [sets[0], validationSet, testSet]


def train(trainingSet, validationSet):
    """Train a series of models with hyper parameter combinations."""
    stats, bestModels = training.trainMNIST(
        trainingSet=trainingSet,
        validationSet=validationSet,
        epochs=100,
        layers=[(170,), (300,), (900,), (300, 300), (900, 100), (170, 100, 70), (300, 200, 100)],
        batchSizes=[16, 32, 128],
        learnRates=[('0.99', lambda epochs, e: 0.99)],
        keepBest=5
    )

    # Save best models and stats:
    for stat in bestModels:
        nn.save(stat[3].topology, 'digits-{}-{}-{:,.2f}'.format(
            stat[0], stat[1], stat[2]
        ))

    with open('digits-stats.json', 'wt') as file:
      json.dump(stats, file)


def test(testSetMNIST):
    """Test a model against MNIST test set and custom images."""
    for path in glob.glob('training/*.model.gz'):
        path.replace(".model.gz", "")

        model = nn.FFN(*nn.load(path.replace(".model.gz", "")))

        # Test accuracy in MNIST:
        accuracy = testing.testMNIST(model, testSetMNIST)
        print("model: {} | MNIST test accuracy: {:,.2%}".format(path, accuracy))

        # Test accuracy with custom images:
        testing.testCustom(
            model,
            ['data/' + i for i in [
                '7_3.png',
                '7_2.png',
                '9_1.png',
                '9_3.png',
                'IMG_3014.jpg',
                'IMG_3015.jpg',
                'digits.jpg']
             ]
        )


if __name__ == '__main__':
    trainingSet, validationSet, testSet = loadMNISTData('data/mnist.dataset.npz')

    # Uncomment to train:
    train(trainingSet, validationSet)

    # Uncomment to test:
    test(testSet)

    # Uncomment to plot a specific model (use filename):
    # statsplot.plotStats("(100, (300,), 16, '0.99')", 'digits-stats.json', 100, [], False)
