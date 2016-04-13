function [ output_args ] = SigDeriv( x )   
    output_args = Sigma(x).*(1 - Sigma(x));
end

