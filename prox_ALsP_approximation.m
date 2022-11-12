function [w] = prox_ALsP_approximation(v, lambda, base)

w = size(v);
total_sum = sum(base.^abs(v));
for i = 1 : length(v)
    w(i) = max(0, abs(v(i)) - lambda * base^abs(v(i)) / total_sum) * sign(v(i));
end
end

