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

/*  This example app demonstrates integrating a neural net classifier
    created in python in an iOS camera app via C++.
 
    1. The model was implemented with NNKit (https://github.com/saldavonschwartz/nnkit)
    and trained on the MNIST dataset, then exported to a file.
 
    2. A forward-only subset of the framework, reimplemented in C++ (nn dir),
    imports and executes the model.
 
    3. Objc / AVFoundation code takes pictures via the phone's camera and sends these
    to a model class, which does additional preprocessing via OpenCV and
    feeds the processed input to the actual neural net model for prediction.

    For more info on the project: https://0xfede.io/2018/05/16/digits.html
 */


#import <UIKit/UIKit.h>
@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property (strong, nonatomic) UIWindow *window;
@end

