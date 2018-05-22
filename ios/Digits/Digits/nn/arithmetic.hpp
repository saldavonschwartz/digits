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

#ifndef arithmetic_hpp
#define arithmetic_hpp

#include "nn.hpp"

namespace nn {
    
    struct Multiply : public NetOp {
        Multiply(NetVar x, NetVar w) :
        NetOp(x.data * w.data, x, w) {}
    };
    
    struct Add : public NetOp {
        Add(const NetVar& x, const NetVar& b) :
        NetOp(x.data + b.data, x, b) {}
    };
}

#endif /* arithmetic_hpp */