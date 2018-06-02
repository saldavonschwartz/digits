//The MIT License (MIT)
//
//Copyright (c) 2018 Federico Saldarini
//https://www.linkedin.com/in/federicosaldarini
//https://github.com/saldavonschwartz
//https://0xfede.io
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/ios.h>
#include <algorithm>
#include "nn.hpp"

#import "NNModel.h"

@implementation NNModelPrediction
@end

@interface NNModel () { nn::Net model;}
@end

@implementation NNModel

- (void)loadModel:(NSString *)filename ready:(void (^)())ready {
    self.modelPath = [[NSBundle mainBundle] pathForResource:filename ofType:@"model.gz"];
    const char* path = [self.modelPath cStringUsingEncoding:NSUTF8StringEncoding];
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        // What takes the most time in loading a model is the string => json convertion...
        self->model = nn::importModel(path);
        self.ready = YES;
        
        dispatch_async(dispatch_get_main_queue(), ready);
    });
}

- (NNModelPrediction*)eval:(CGImageRef)image roi:(CGRect)roi {
    assert(self.ready);
    auto img = CGImageRef2CVMat(image);
    
    // Crop to ROI if needed:
    if (!CGRectEqualToRect(roi, CGRectZero)) {
        cv::Rect roiBounds(
             round(roi.origin.x * img.cols),
             round(roi.origin.y * img.rows),
             round(roi.size.width * img.cols),
             round(roi.size.height * img.rows)
         );
        
        img = img(roiBounds);
    }
    
    cv::Mat binary, edges;
    auto contours = extractContours(img, binary, edges, self.thresholdMode);
    
    if (!contours.size()) {
        return nil;
    }
    
    // Make a copy of binary image for annotation / display:
    cv::Mat binaryCopy;
    cv::cvtColor(binary, binaryCopy, cv::COLOR_GRAY2BGR);
    
    // Use  first sub-ROI:
    auto contour = contours[0];
    auto b = cv::boundingRect(contour);
    auto subroi = binary(b);
    
    // Normalize:
    subroi = normalize(subroi);
    
    // Make prediction:
    std::vector<float> modelIn;
    auto roiFlat = subroi.clone().reshape(1,1);
    roiFlat /= 255;
    roiFlat.copyTo(modelIn);
    
    int p = 0;
    auto modelOut = model({{modelIn}});
    auto confidence = modelOut.row(0).maxCoeff(&p);
    
    // Annotate image copy:
    cv::rectangle(binaryCopy, {b.x, b.y}, {b.x+b.width, b.y+b.height}, {255, 120, 0, 255}, 5);
    
    // Create prediction object:
    NNModelPrediction* prediction = [NNModelPrediction new];
    prediction.image = MatToUIImage(binaryCopy);
    prediction.roi = MatToUIImage(subroi);
    prediction.confidence = confidence;
    prediction.value = p;
    
    return prediction;
}

cv::Mat CGImageRef2CVMat(CGImageRef cgImage) {
    int w = (int)CGImageGetWidth(cgImage);
    int h = (int)CGImageGetHeight(cgImage);
    CGColorSpaceRef cspace = CGImageGetColorSpace(cgImage);
    cv::Mat cvImage(w, h, CV_8UC4);
    
    CGContextRef ctx =
    CGBitmapContextCreate(cvImage.data, h, w, 8, cvImage.step[0], cspace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault);
    
    // Rotate image:
    CGContextRotateCTM (ctx, -M_PI_2);
    CGContextTranslateCTM(ctx, -w, 0);
    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), cgImage);
    CGContextRelease(ctx);
    return cvImage;
}

std::vector<std::vector<cv::Point>> extractContours(const cv::Mat& colorIn, cv::Mat& binaryOut, cv::Mat& edgesOut, ThresholdMode thresholdMode) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    // Filter out some high freqs:
    cv::cvtColor(colorIn, binaryOut, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(binaryOut, binaryOut, {11, 11}, 0);
    
    // Increase contrast:
    binaryOut.convertTo(binaryOut, CV_64FC4);
    for (int i = 0;  i < binaryOut.rows; i++) {
        for (int j = 0;  j < binaryOut.cols; j++) {
            binaryOut.at<double>(i, j) = MIN(binaryOut.at<double>(i, j) * 2. - 255. , 255.);
            binaryOut.at<double>(i, j) = MAX(binaryOut.at<double>(i, j), 0.);
        }
    }
    binaryOut.convertTo(binaryOut, CV_8UC4);
    
    // Extract contours from binary edges:
    if (thresholdMode == ThresholdModeFixed) {
        cv::threshold(255 - binaryOut, binaryOut, 220, 255, cv::THRESH_BINARY);
    }
    else {
        cv::threshold(255 - binaryOut, binaryOut, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }
    
    cv::Canny(binaryOut, edgesOut, 50, 255);
    cv::findContours(edgesOut, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    if (contours.size()) {
        // Sort by decreasing bounds area:
        std::sort(contours.begin(), contours.end(), [](auto& a, auto& b) {
            return cv::boundingRect(a).area() > cv::boundingRect(b).area();
        });
    }
    
    return contours;
}

cv::Mat normalize(const cv::Mat& binaryIn) {
    cv::Mat k{5,5};
    cv::Mat normOut;
    cv::dilate(binaryIn, normOut, k);
    
    int diff = abs(normOut.rows - normOut.cols);
    int p1, p2;
    
    if (!diff % 2) {
        p1 = diff/2;
        p2 = p1;
    }
    else {
        p1 = diff;
        p2 = 0;
    }
    
    // Center around 20x20 square, with 4px padding on each side:
    if (normOut.cols < normOut.rows) {
        cv::copyMakeBorder(normOut, normOut, 0, 0, p1, p2, cv::BORDER_CONSTANT, {0});
    }
    else if (normOut.cols > normOut.rows) {
        cv::copyMakeBorder(normOut, normOut, p1, p2, 0, 0, cv::BORDER_CONSTANT, {0});
    }
    
    cv::resize(normOut, normOut, {20, 20});
    cv::copyMakeBorder(normOut, normOut, 4, 4, 4, 4, cv::BORDER_CONSTANT, {0});
    
    // Center of mass:
    auto m = cv::moments(normOut);
    auto cx = m.m10 / m.m00;
    auto cy = m.m01 / m.m00;
    auto sx = round(normOut.cols / 2. - cx);
    auto sy = round(normOut.rows / 2. - cy);
    cv::Mat T = (cv::Mat_<float>(2, 3) << 1, 0, sx, 0, 1, sy);
    cv::warpAffine(normOut, normOut, T, normOut.size());
    cv::dilate(normOut, normOut, k);
    return normOut;
}

@end
