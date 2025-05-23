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
        other->grad += this->data * out.grad;
    };

    return out;
}


Value Value::pow(float other) {
    std::string op = "**{}" + std::to_string(other);
    Value out = Value(std::pow(this->data, other), {*this}, op);
    out._backward = [this, &other, &out](){
        this->grad += (other * std::pow(this->data, other-1)) * out.grad;
    };
    return out;
}

Value Value::relu() {
    Value out = Value(0, {*this}, "ReLU");
    out.data = (this->data < 0) ? 0: this->data;

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
    build_topo(this);
    this->grad = 1;
    for(auto v = std::rbegin(topo); v != std::rend(topo); v++) {
        v->_backward();
    }
};

bool Value::operator==(const Value *other) const {
    return this == other;
}

bool operator==(Value &self, Value &other) {
    return &self == &other;
}

Value operator-(Value &self)  {
   return (-1) * self;
}

// plus ops
Value operator+(Value &self, Value &other) {
    return self.add(&other);
}

Value operator+(Value &self, float other) {
    Value other_val = Value(other, {}, "");
    return self + other_val;
}

Value operator+(float other, Value &self) {
    Value other_val = Value(other, {}, "");
    return other_val + self;
}

Value operator+(Value self, const Value &other) {
    return self.add(const_cast<Value*>(&other));
}

Value operator-(Value &self, Value &other) {
    Value other_val = Value(-other.data, {}, "");
    return self.add(&other_val);
}

Value operator-(Value &self, float other) {
    Value other_val = Value(other, {}, "");
    return self - other_val;
}

Value operator-(float other, Value &self) {
    Value other_val = Value(other, {}, "");
    return other_val - self;
}

Value operator-(Value self, const Value &other) {
    Value other_val = Value(-other.data, {}, "");
    return self.add(const_cast<Value*>(&other_val));
}

Value operator*(Value &self, Value &other) {
    return self.mul(&other);
}

Value operator*(Value &self, float other) {
    Value tmp = Value(other, {}, "");
    return self.mul(&tmp);
}

Value operator*(float other, Value &self) {
    Value tmp = Value(other, {}, "");
    return tmp.mul(&self);
}

Value operator*(Value self, const Value &other) {
    return self.mul(const_cast<Value*>(&other));
}

Value operator/(Value &self, Value &other) {
    Value other_val = other.pow(-1);
    return self * other_val;
}

Value operator/(Value &self, float other) {
    Value other_val = Value(other, {}, "");
    return self / other_val;
}

Value operator/(float other, Value &self) {
    Value other_val = Value(other, {}, "");
    return other_val / self;
}

Value operator/(Value self, const Value &other) {
    Value other_val = Value(std::pow(other.data, -1), {}, "");
    return self.mul(const_cast<Value*>(&other_val));
}

// ^ was used as power op.
Value operator^(Value self, float other) {
    std::cout << self.data << std::endl;
    return self.pow(other);
}

int main(int argc, char **argv) {
    auto a = Value(-4.0, {}, "");
    auto b = Value(2.0, {}, "");
    auto c = a + b;
    auto d = a * b + (b^3.0);  // -8 + 8 = 0
    std::cout << d.data << std::endl;
    c = c + c + 1;
    c = c + 1 + c + (-a);
    d = d + d * 2 + (b + a).relu();
    d = d + 3*d + (b - a).relu();
    auto e = c - d;
    auto f = e^2.0;
    auto g = f / 2.0;
    g = g + 10.0 / f;
    std::cout << "g.data:" << g.data << std::endl;
    g.backward();
    return 0;
}