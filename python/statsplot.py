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

import matplotlib.pyplot as plt
import numpy as np
import json
import re


def plotStats(modelPath, statsPath):
    """Plot training / validation loss plus validation accuracy for a model"""

    match = re.match(r".*-(\(\d+, \(.*\))-(\d+).*", modelPath)
    key = match.group(1)

    with open(statsPath, 'rt') as file:
        stats = json.load(file)[key]

    title = key.replace("(", "").replace(")", "").replace("'", "").replace(",", "-").replace(" ", "").replace("--", "-")
    epochRange = list(range(1, 101))
    trainLoss = [stats[str(e)][0] for e in epochRange]
    valLoss = [stats[str(e)][1] for e in epochRange]
    valAccu = [stats[str(e)][2] for e in epochRange]
    lowValLoss = [np.argmin(valLoss) + 1], [np.min(valLoss)]
    highValAccu = [np.argmax(valAccu) + 1], [np.max(valAccu)]

    figure = plt.figure(figsize=(9, 6))
    ax = figure.add_subplot(1, 1, 1)
    ax.set_xlim(1, 100)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_ylim(0, 0.2)
    ax.autoscale(enable=True, axis='y', tight=None)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    p1 = ax.plot(
        epochRange, trainLoss,
        label='training loss', color=(237/255, 102/255, 93/255)
    )[0]

    p2 = ax.plot(
        epochRange, valLoss,
        label='validation loss', color=(255/255, 158/255, 74/255)
    )[0]

    ax.scatter(*lowValLoss, s=120, alpha=0.5)
    ax.annotate(
        s='epoch: {}\nloss: {:,.3f}'.format(lowValLoss[0][0], lowValLoss[1][0]),
        xy=(lowValLoss[0][0], lowValLoss[1][0] - 0.01), fontsize=7
    )

    ax2 = ax.twinx()
    ax2.set_ylim(90, 100)
    ax2.autoscale(enable=True, axis='y', tight=None)
    ax2.set_ylabel('accuracy (%)')

    p3 = ax2.plot(
        epochRange, valAccu,
        label='validation accuracy', color=(109/255, 204/255, 218/255)
    )[0]

    ax2.scatter(*highValAccu, s=120, alpha=0.5)
    ax2.annotate(
        s='epoch: {}\naccuracy: {:,.2f}%'.format(highValAccu[0][0], highValAccu[1][0]),
        xy=(highValAccu[0][0], highValAccu[1][0] + 0.01), fontsize=7
    )

    ax.legend(
        [p1, p2, p3], [p1.get_label(), p2.get_label(), p3.get_label()],
        loc='lower center', frameon=False, bbox_to_anchor=(0, 1, 1., 1), ncol=3
    )

    plt.title('MNIST Digit Classifier: {}\n\n'.format(title))
    plt.tight_layout()
    plt.show(block=True)
