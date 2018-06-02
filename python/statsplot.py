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


def markEpoch(epoch, epochs, stats, ax1, ax2, best):
    size = 120 if best else 70
    alpha = 0.5 if best else 0.3

    stat = stats[str(epoch)]

    ax2.scatter([epoch], [stat[2]], s=size, alpha=alpha, color=(109 / 255, 204 / 255, 218 / 255))
    ax2.annotate(
        s='{:,.2f} {}'.format(stat[2], '*BEST*' if best else ""),
        xy=(epoch + 0.005, stat[2] + 0.005), fontsize=7
    )

    ax1.scatter([epoch], [stat[1]], s=size, alpha=alpha, color=(255 / 255, 158 / 255, 74 / 255))
    ax1.annotate(
        s='{:,.3f}'.format(stat[1]),
        xy=(epoch + 0.005, stat[1] + 0.005), fontsize=7
    )

    ax1.scatter([epoch], [stat[0]], s=size  , alpha=alpha, color=(237 / 255, 102 / 255, 93 / 255))
    ax1.annotate(
        s='{:,.3f}'.format(stat[0]),
        xy=(epoch + 0.005, stat[0] + 0.005), fontsize=7
    )

    ax1.set_xticks(list(ax1.get_xticks()) + [epoch])
    ax2.set_xticks(list(ax2.get_xticks()) + [epoch])
    ax1.set_xlim(1, epochs)
    ax2.set_xlim(1, epochs)
    ax1.axvline(x=epoch, linewidth=0.5, alpha=alpha)


def plotStats(key, statsPath, epochs, extraMarkEpochs, save):
    """Plot training / validation loss plus validation accuracy for a model"""
    with open(statsPath, 'rt') as file:
        stats = json.load(file)[key]

    match = re.match(r"\(\d+, \((.*)\), (\d+), (.*)\).*", key)
    topology = match.group(1).replace(" ", "-").replace(",", "")
    batchSize = match.group(2)
    learnRate = match.group(3).replace("'", "")

    epochRange = list(range(1, epochs+1))
    trainLoss = [stats[str(e)][0] for e in epochRange]
    valLoss = [stats[str(e)][1] for e in epochRange]
    valAccu = [stats[str(e)][2] for e in epochRange]
    lowValLoss = [np.argmin(valLoss) + 1], [np.min(valLoss)]
    highestValAccu = np.argmax(valAccu) + 1

    figure = plt.figure(figsize=(9, 6))
    ax = figure.add_subplot(1, 1, 1)
    ax.set_xlim(1, epochs)
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

    ax.scatter(*lowValLoss, s=120, alpha=0.5, color=(255 / 255, 158 / 255, 74 / 255))
    ax.annotate(
        s='{:,.3f} -LOWEST-'.format(lowValLoss[1][0]),
        xy=(lowValLoss[0][0] + 0.005, lowValLoss[1][0] + 0.005), fontsize=7
    )

    ax2 = ax.twinx()
    ax2.set_ylim(90, 100)
    ax2.autoscale(enable=True, axis='y', tight=None)
    ax2.set_ylabel('accuracy (%)')

    p3 = ax2.plot(
        epochRange, valAccu,
        label='validation accuracy', color=(109/255, 204/255, 218/255)
    )[0]

    markEpoch(highestValAccu, epochs, stats, ax, ax2, True)
    for epoch in extraMarkEpochs:
        markEpoch(epoch, epochs, stats, ax, ax2, False)

    ax.legend(
        [p1, p2, p3], [p1.get_label(), p2.get_label(), p3.get_label()],
        loc='lower center', frameon=False, bbox_to_anchor=(0, 1, 1., 1), ncol=3
    )

    plt.title('Digits Classifier Stats:\ntopology:{} | batch size: {} | learn rate: {}\n\n'.format(
        topology, batchSize, learnRate
    ), fontsize=10)

    plt.tight_layout()

    if save:
        plt.savefig(
            'plot-t{}-b{}-l{}.svg'.format(topology, batchSize, learnRate),
            format='svg', dpi=1200, transparent=True
        )
    else:
        plt.show(block=True)
