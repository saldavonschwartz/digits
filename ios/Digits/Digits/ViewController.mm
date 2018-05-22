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

#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import "NNModel.h"

#if TARGET_IPHONE_SIMULATOR
#define TEST_IMG @"IMG_3015.JPG"
#endif

@interface ViewController () <AVCapturePhotoCaptureDelegate>
@property (weak, nonatomic) IBOutlet UIView *previewView;
@property (weak, nonatomic) IBOutlet UIView *targetView;
@property (weak, nonatomic) IBOutlet UIButton *captureButton;
@property (weak, nonatomic) IBOutlet UILabel *predictionLabel;
@property NNModel* model;
@property AVCapturePhotoOutput* camOut;
@property CGRect ROI;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.predictionLabel.text = @"loading model...";
    self.previewView.userInteractionEnabled = NO;
    self.previewView.alpha = 0;
    
    // Load model:
    self.model = [NNModel new];
    self.model.thresholdMode = ThresholdModeFixed;
    [self.model loadModel:@"digits" ready:^{
        self.predictionLabel.text = nil;
        [UIView animateWithDuration:.5 animations:^{
            self.previewView.userInteractionEnabled = YES;
            self.previewView.alpha = 1;
        }];
    }];
    
    self.targetView.backgroundColor = [UIColor clearColor];
    self.targetView.layer.borderColor = [UIColor colorWithRed:1. green:120./255. blue:0 alpha:1.].CGColor;
    self.targetView.layer.cornerRadius = 8;
    self.targetView.layer.borderWidth = 4;
    
    // Compute ROI (region of interest) as target
    // view percentage of preview view:
    CGRect f1 = self.targetView.frame;
    CGRect f2 = self.previewView.frame;
    self.ROI = {
        f1.origin.x / f2.size.width,
        f1.origin.y / f2.size.height,
        f1.size.width / f2.size.width,
        f1.size.height / f2.size.height
    };
    
    [self setupCaptureSession];
}

- (void)setupCaptureSession {
#if TARGET_IPHONE_SIMULATOR
    UIImage* img = [UIImage imageNamed:TEST_IMG];
    UIImageView* imgView = [[UIImageView alloc]initWithImage:img];
    imgView.frame = self.previewView.bounds;
    [self.previewView.layer insertSublayer:imgView.layer atIndex:0];
#else
    NSArray* devices = [AVCaptureDeviceDiscoverySession
                        discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera]
                        mediaType:AVMediaTypeVideo
                        position:AVCaptureDevicePositionBack].devices;
    
    AVCaptureDevice* camera = devices.firstObject;
    [camera lockForConfiguration:nil];
    camera.focusMode = AVCaptureFocusModeContinuousAutoFocus;
    [camera unlockForConfiguration];
    
    AVCaptureSession* session = [AVCaptureSession new];
    AVCaptureDeviceInput* input = [AVCaptureDeviceInput deviceInputWithDevice:camera error:nil];
    [session addInput:input];
    
    self.camOut = [AVCapturePhotoOutput new];
    [session addOutput:self.camOut];
    [session startRunning];
    
    AVCaptureVideoPreviewLayer* previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:session];
    previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
    previewLayer.connection.videoOrientation = AVCaptureVideoOrientationPortrait;
    [self.previewView.layer insertSublayer:previewLayer atIndex:0];
    previewLayer.frame = self.previewView.bounds;
#endif
}

- (IBAction)onCapture:(UIButton *)sender {
#if TARGET_IPHONE_SIMULATOR
    UIImage* img = [UIImage imageNamed:TEST_IMG];
    [self processInput:img.CGImage];
#else
    AVCapturePhotoSettings* settings = [AVCapturePhotoSettings photoSettings];
    [self.camOut capturePhotoWithSettings:settings delegate:self];
#endif
}

- (void)captureOutput:(AVCapturePhotoOutput *)output didFinishProcessingPhoto:(nonnull AVCapturePhoto *)photo error:(nullable NSError *)error {
    [self processInput:photo.CGImageRepresentation];
}

- (void)processInput:(CGImageRef)input {
    NNModelPrediction* prediction = [self.model eval:input roi:self.ROI];
    
    if (!prediction) {
        return;
    }
    
    // Disable image capture while displaying prediction:
    self.predictionLabel.text = [NSString stringWithFormat:@"%zi", prediction.value];
    self.captureButton.enabled = NO;
    
    UIImageView* view = [[UIImageView alloc] initWithImage:prediction.image];
    view.frame = self.targetView.bounds;
    
    [UIView animateWithDuration:.2 animations:^{
        self.predictionLabel.alpha = 1;
        self.captureButton.alpha = 0.;
        [self.targetView.layer addSublayer:view.layer];
        self.previewView.layer.sublayers[0].opacity = .1;
    } completion:^(BOOL finished) {
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
            // Reenable image capture:
                [UIView animateWithDuration:.2 animations:^{
                    self.predictionLabel.alpha = 0;
                    self.previewView.layer.sublayers[0].opacity = 1.;
                    [view.layer removeFromSuperlayer];
                    self.captureButton.alpha = 1;
                } completion:^(BOOL finished) {
                    self.predictionLabel.text = nil;
                    self.captureButton.enabled = YES;
                }];
        });
    }];
}

@end
