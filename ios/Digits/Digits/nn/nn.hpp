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

#ifndef nn_hpp
#define nn_hpp

#include <functional>
#include <vector>
#include "3rdparty/Eigen/Core"

namespace nn {
    using std::vector;
    using std::function;
    using vec = vector<vector<float>>;
    using ndarray = Eigen::MatrixXf;
    
    struct NetVar {
        ndarray data;
        NetVar(const ndarray& data) : data(data) {}
        NetVar(const vec& data) : data(data.size(), data[0].size()) {
            for (int i = 0; i < data.size(); i++) {
                for (int j = 0; j < data[0].size(); j++) {
                    this->data(i, j) = data[i][j];
                }
            }
        }
    };
    
    struct NetOp : public NetVar {
        vector<const NetVar*> parents;
        template <class... P>
        NetOp(const ndarray& data, const P&... parents) : NetVar(data), parents({&parents...}) {}
    };
    
    struct LayerBase {
        function<NetOp(const NetVar& x)> eval;
    };
    
    template <class Op>
    struct Layer : public LayerBase {
        template <class... P>
        Layer(const P&... parents) {
            eval = [parents...](const NetVar& x) {
                return Op(x, parents...);
            };
        }
    };
    
    struct Net {
        vector<LayerBase> topology;
        Net() = default;
        Net(const vector<LayerBase>& topology) : topology(topology) {}
        
        ndarray operator()(NetVar x) {
            for (auto& layer : topology) {
                x = layer.eval(x);
            }
            
            return x.data;
        }
    };
    
}

#include "arithmetic.hpp"
#include "activation.hpp"
#include "serialization.hpp"

#endif /* nn_hpp */


