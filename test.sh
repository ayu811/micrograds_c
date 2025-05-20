#/bin/bash

g++ -std=c++17 -c engine.cc -o engine.o
g++ -std=c++17 -c nn.cc -o nn.o
g++ -std=c++17 nn.o engine.o -o nn