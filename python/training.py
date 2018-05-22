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

from itertools import product
from collections import deque
import nnkit as nn
import numpy as np


def trainMNIST(trainingSet, validationSet, epochs, layers, batches, learnRate, keepBest):
    """Train and validate models on MNIST with different hyper parameter combinations.

    This function trains and validates one model per hyper parameter combination. It then returns
    all training stats and the n (keepBest) models that scored the highest validation accuracy.

    :param trainingSet: a 2-tuple where the first element contains the MNIST training set examples and
    the second element contains the target labels for each example in one-hot form.

    :param validationSet: a 2-tuple where the first element contains MNIST examples and
    the second element contains the target labels for each example in one-hot form.
    This set must be disjoint from the training set.

    :param epochs: a list of epochs to train for.
    i.e.: [10, 50, 100]

    :param layers: a list of tuples where each element represents the (hidden unit) size of a layer.
    i.e.: [(10,), (10,20)]

    :param batches: a list of batch sizes to use when training.
    i.e.: [16, 32]

    :param learnRate: a list of 2-tuples where the first element is an arbitrary string id and the second element
    is a lambda taking the total number of epochs and the current epoch and returning the learning rate for that epoch.
    i.e.: [('fixed 0.4', lambda epochs, e: 0.4), ('decay-0.4-0.1', lambda epochs, e: nn.decay(e, epochs, (0.1, 0.4)))]

    :param keepBest: How many of the best models to return.

    :return: a 2-tuple where the first element is a dictionary of training stats for all models and the second element
    is a list of the highest n (keepBest) scoring models.
    """
    # x = data in (raw pixels), y = one-hot target for loss evaluation:
    x, y = nn.NetVar(), nn.NetVar()

    combinations = product(epochs, layers, batches, learnRate)
    validationTarget = np.argmax(validationSet[1].data, axis=1)
    bestQueue = deque(maxlen=keepBest)
    allStats = {}
    best = None

    # Train one model per hyper parameter combination:
    for epochs, layers, batches, learnRate in combinations:
        key = str((epochs, layers, batches, learnRate[0]))
        stats = {}
        allStats[key] = stats

        print('++ NEW MODEL: {} ++'.format(key))

        # Define model topology according to layers hyper param:
        topology = []

        for i in range(len(layers)):
            topology.extend([
                (nn.Multiply, nn.rand2(28 * 28 if not i else layers[i-1], layers[i])),
                (nn.Add, nn.rand2(layers[i])),
                (nn.ReLU,)
            ])

        topology.extend([
            (nn.Multiply, nn.rand2(layers[i], 10)),
            (nn.Add, nn.rand2(10)),
            (nn.SoftMax,)
        ])

        net = nn.FFN(*topology)

        # Create optimizer and loss node:
        optimizer = nn.GD(net.vars)
        net.topology.append((nn.CELoss, y))

        for e in range(1, epochs+1):
            abort = False
            optimizer.learnRate = learnRate[1](epochs, e-1)

            # Train:
            for inputs, targets, i in nn.miniBatch(trainingSet, size=batches):
                x.data, y.data = inputs, targets
                trainingLoss = net(x)

                if trainingLoss == float('inf'):
                    print('e: {} | batch: {}. Vanishing gradient: abort...'.format(e, i + 1))
                    abort = True
                    break

                net.back()
                optimizer.step()

            if abort:
                break

            # Validate:
            x.data, y.data = validationSet[0], validationSet[1]
            validationLoss = net(x)

            # Prediction is output of antepenultimate layer,
            # because last layer is loss node during training:
            prediction = net.layers[-2].data
            prediction = np.argmax(prediction, axis=1)
            validationAccuracy = np.mean(prediction == validationTarget) * 100
            stats[e] = (trainingLoss.item(), validationLoss.item(), validationAccuracy)

            # Keep best model so far:
            newBest = False
            if not len(bestQueue) or bestQueue[-1][2] < validationAccuracy:
                topology = [
                    [nn.NetVar(np.copy(n.data)) if type(n) is nn.NetVar else n for n in layer]
                    for layer in net.topology[:-1]
                ]

                best = (key, e, validationAccuracy, nn.FFN(*topology))
                bestQueue.append(best)
                newBest = True

            print('\t e:{} | train loss: {:,.3f} | val loss: {:,.3f} | val accuracy: {:,.2f}% | learn rate: {:,.2f} {}'.format(
                e, *stats[e], optimizer.learnRate, " *" if newBest else ""
            ))

    return allStats, bestQueue
