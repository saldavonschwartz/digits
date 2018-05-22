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

#ifndef serialization_hpp
#define serialization_hpp

#include <zlib.h>
#include <string>
#include "3rdparty/json.hpp"
#include "nn.hpp"
#include <chrono>

struct profiler {
    std::string name;
    std::chrono::high_resolution_clock::time_point p;
    
    profiler(std::string const &n) :
    name(n), p(std::chrono::high_resolution_clock::now()) { }
    
    ~profiler() {
        using duration = std::chrono::duration<double>;
        auto d = std::chrono::high_resolution_clock::now() - p;
        std::cout << name << ": "
        << std::chrono::duration_cast<duration>(d).count()
        << " sec."
        << std::endl;
    }
};

#define PROFILE_BLOCK(pbn) profiler _pfinstance(pbn)

namespace nn {
    using jsn = nlohmann::json;
    
    Net importModel(const char* filename) {
        std::cout << "\n---- importing model -----\n" << filename << "\n";
        {
            PROFILE_BLOCK("---- done -----");
            
            int bufferSize = 256;
            gzFile file;
            
            {
                PROFILE_BLOCK("open zip\t\t\t");
                file = gzopen(filename,"rb");
            }
            
            char* buffer = new char[bufferSize];
            std::string data;
            
            {
                PROFILE_BLOCK("zip >> memory\t");
                while(gzgets(file, buffer, bufferSize))
                    data += buffer;
            }
            
            jsn json;
            
            {
                PROFILE_BLOCK("memory >> json\t");
                json = jsn::parse(data);
            }
            
            std::vector<nn::LayerBase> topology;
            
            {
                PROFILE_BLOCK("json >> net\t\t");
                
                for (auto& node : json) {
                    auto opType = node["op"].get<std::string>();
                    
                    if (opType == "Multiply") {
                        auto w = node["args"][0].get<std::vector<std::vector<float>>>();
                        topology.push_back(nn::Layer<nn::Multiply>(w));
                    }
                    else if (opType == "Add") {
                        auto w = node["args"][0].get<std::vector<std::vector<float>>>();
                        topology.push_back(nn::Layer<nn::Add>(w));
                    }
                    else if (opType == "ReLU") {
                        topology.push_back(nn::Layer<nn::ReLU>());
                    }
                    else if (opType == "SoftMax") {
                        topology.push_back(nn::Layer<nn::SoftMax>());
                    }
                }
            }
            
            return nn::Net(topology);
        }
    }
}

#endif /* serialization_hpp */
