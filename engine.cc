#include <iostream>
#include "engine.hpp"


Value::Value(float data, std::vector<Value> children, std::string op) : data(data), _prev(children), _op(op), grad(0.0f) {};

Value Value::add(Value *other) {
    Value out = Value(0, {*this, *other}, "+");
    out.data = this->data + other->data;
    out._backward = [this, &other, &out](){
        this->grad += out.grad;
        other->grad += out.grad;
    };

    return out;
}

Value Value::mul(Value *other) {
    Value out = Value(0, {*this, *other}, "*");
    out.data = this->data * other->data;
    out._backward = [this, &other, &out]() {
        this->grad += other->data * out.grad;
        other->grad += this->grad * out.grad;
    };

    return out;
}


Value Value::pow(float other) {
    std::string op = "**{}" + std::to_string(other);
    Value out = Value(0, {*this}, op);
    out._backward = [this, &other, &out](){
        this->grad += (other * std::pow(this->data, other-1)) * out.grad;
    };
    return out;
}

Value Value::relu() {
    Value out = Value(0, {*this}, "ReLU");
    if(out.data < 0)
        this->data = 0;
    else
        out.data = this->data;

    out._backward = [this, &out]() {
        this->grad = (out.data <= 0) ? 0 : out.grad;
    };
    return out;
}

void Value::backward() {
    std::vector<Value> topo;
    std::vector<Value> visited;

    std::function<void(Value*)>build_topo = [&topo, &visited, &build_topo](Value *v) {
        bool in_vec = std::find(visited.begin(), visited.end(), v) != visited.end();
        if(in_vec) {
            visited.push_back(*v);
            for(auto&& child: v->_prev)
                build_topo(&child);
            topo.push_back(*v);
        }
    };
    this->grad = 1;
    for(auto v = std::rbegin(topo); v != std::rend(topo); v++) {
        v->_backward();
    }
};

bool Value::operator==(const Value *other) const {
    return this == other;
}
Value operator+(Value &self, Value &other) {
    return self.add(&other);
}
int main(int argc, char **argv) {
    Value a = Value(1, {}, "");
    Value b = Value(2, {}, "");
    auto c = a + b;
    c.backward();
    std::cout << c.data << std::endl;
    return 0;
}