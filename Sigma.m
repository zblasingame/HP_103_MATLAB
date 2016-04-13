function [ output_args ] = Sigma( x )
    output_args = 1 / (1 + exp(-x));
end

