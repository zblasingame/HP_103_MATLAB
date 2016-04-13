function [ output_args ] = LReLUD( x )   
    if x <= 0
        output_args = 0.001;
    else
        output_args = 1;
    end
end

