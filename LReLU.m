function [ output_args ] = LReLU( x )
    if x <= 0
       output_args = 0.001*x;
    else
       output_args = x;
    end
end

