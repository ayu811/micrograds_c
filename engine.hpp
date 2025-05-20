#ifndef ENGINE_H_
#define ENGINE_H_
#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <algorithm>

class Value {
    public:
        Value(float data, std::vector<Value> children = {}, std::string op = "");
        float data;
        float grad;
        std::function<void(void)> _backward;
        std::vector<Value> _prev;
        std::string _op;

        Value add(Value *other);
        Value mul(Value *other);
        Value pow(float other);
        Value relu();

        void backward(void);

        bool operator==(const Value *other) const;
    };

Value operator+(Value &self, Value &other);

#endif