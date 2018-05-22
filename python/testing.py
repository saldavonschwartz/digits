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

import nnkit as nn
import numpy as np
import cv2 as cv


def testMNIST(model, testSet):
    """Test a model against the MNIST test set.
    
    :param model: a model to test on. 
    
    :param testSet: a 2-tuple where the first element contains the MNIST test examples and 
    the second element contains the target labels for each example in one-hot form.
    The test set needs to be disjoint from BOTH the training and validation sets.
     
    :return: a test accuracy score, as a percentage.
    """
    x = nn.NetVar(testSet[0])
    prediction = model(x)
    prediction = np.argmax(prediction, axis=1)
    target = np.argmax(testSet[1], axis=1)
    accuracy = np.mean(prediction == target)
    return accuracy


def testCustom(model, imgPaths):
    """Test a model against custom images.

    :param model: model to test on.
    :param images: a list of paths to images.
    """

    for path in imgPaths:
        # Extract all detected contours from image:
        img = cv.imread(path, cv.CV_8UC4)
        contours, binary, _ = extractContours(img)

        # Create a copy if the image for annotation / display:
        binaryCopy = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

        for contour in contours:
            # Extract ROI and normalize it:
            x, y, w, h = cv.boundingRect(contour)
            roi = binary[y:y + h, x:x + w]
            roi = normalize(roi)

            # Make prediction:
            modelIn = roi.flatten() / 255
            modelOut = model(modelIn)
            prediction = np.argmax(modelOut)

            # Annotate image copy:
            R = np.random.randint(0, 255)
            G = np.random.randint(0, 255)
            B = np.random.randint(0, 255)
            color = (B, G, R, 255)

            cv.rectangle(binaryCopy, (x, y), (x + w, y + h), color, 4)
            cv.putText(binaryCopy, str(prediction), (x + w // 2 - 10, y + h // 2 + 10), 2, 1, color, 1)

        # Display annotated image:
        cv.namedWindow(path, cv.WINDOW_NORMAL)
        cv.imshow(path, binaryCopy)
        cv.waitKey()
        cv.destroyAllWindows()


def extractContours(colorIn):
    """Extract contours of potential ROIs from an image.

    :param colorIn: a 3-channel color image (ndarray) to find contours in.
    
    :return: (contours, binaryOut, edges), where:
    . contours: a list of 2d point sets making up the contour of each ROI.
    . binaryOut: a binary inverted (black and white) thresholded copy of colorIn.
    . a binary edge version of binaryOut.
    """
    contours = []

    # Filter out some high freqs:
    binaryOut = cv.cvtColor(colorIn, cv.COLOR_BGR2GRAY)
    binaryOut = cv.GaussianBlur(binaryOut, (11, 11), 0)

    # Increase contrast:
    binaryOut = np.double(binaryOut)

    for i in range(binaryOut.shape[0]):
        for j in range(binaryOut.shape[1]):
            binaryOut[i][j] = min(binaryOut[i][j] * 1.9 - 255., 255.)
            binaryOut[i][j] = max(binaryOut[i][j], 0.)

    binaryOut = np.uint8(binaryOut)

    # Extract contours from binary edges:
    _, binaryOut = cv.threshold(255 - binaryOut, 180, 255, cv.THRESH_BINARY)
    edges = cv.Canny(binaryOut, 50, 255)
    _, contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours, binaryOut, edges


def normalize(binaryIn):
    """Center an image inside a 20x20 px area and pads it by 4px on each side.

    :param binaryIn: a binary image.
    :return: a normalized 28x28 px version of binaryIn.
    """
    k = 5, 5
    normOut = cv.dilate(binaryIn, k)
    rows, cols = normOut.shape

    diff = abs(rows - cols)
    p1, p2 = (diff // 2, diff // 2) if diff % 2 is 0 else (diff, 0)

    # Center around 20x20 square, with 4px padding on each side:
    if cols < rows:
        normOut = cv.copyMakeBorder(normOut, 0, 0, p1, p2, cv.BORDER_CONSTANT, (0,))
    elif cols > rows:
        normOut = cv.copyMakeBorder(normOut, p1, p2, 0, 0, cv.BORDER_CONSTANT, (0,))

    normOut = cv.resize(normOut, (20, 20))
    normOut = cv.copyMakeBorder(normOut, 4, 4, 4, 4, cv.BORDER_CONSTANT)

    # Center of mass:
    m = cv.moments(normOut)
    if m['m00'] == 0:
        cx, cy = 0, 0
    else:
        cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']

    rows, cols = normOut.shape
    sx = np.round(cols / 2. - cx).astype(int)
    sy = np.round(rows / 2. - cy).astype(int)

    T = np.float32([[1, 0, sx], [0, 1, sy]])
    normOut = cv.warpAffine(normOut, T, normOut.shape)
    normOut = cv.dilate(normOut, k)
    return normOut
