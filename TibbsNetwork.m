classdef TibbsNetwork < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        filename
        MAX_ITERATIONS
        learningCoefficient
        
        numInputs
        numNeurons
        numCases
        
        % each row is a particular input, columns are the various cases
        trainingInputs
        % Column with each row of size 1 being one case's output
        idealOutputs
        
        % Between the initial inputs and hidden layer
        % Each column is for one neuron, the number of rows is numInputs+1
        hweights
        % Between the hidden layer and the output
        % 1 column, number of rows is numInputs+1
        oweights
        
        % number of columns are the iterations, rows are the various
        %   cases
        errors
    end
    
    methods
        % Constructor
        function this = TibbsNetwork(numInputs_, numNeurons_, maxIts_, filename_, learningco_)
            % Set properties
            this.numInputs = numInputs_;
            this.numNeurons = numNeurons_;
            this.MAX_ITERATIONS = maxIts_;
            this.learningCoefficient = learningco_;
            
            
            % Load file
            FileContents = importdata(filename_, '|', 1);
            
            % Read Training inputs (flipped in the file)
            for i=1:1:this.numInputs
                this.trainingInputs(i, :) = FileContents.data(:,i);
            end
            
            %Figure out how many cases there (IE number of columns)
            [~, this.numCases] = size(this.trainingInputs);
            
            % Read ideal outputs, the last column of the file
            this.idealOutputs = FileContents.data(:, this.numInputs+1);
            
            
            % Make random weights for the hidden layer and output
            this.hweights = rand(this.numInputs+1, this.numNeurons);
            this.oweights = rand(this.numNeurons+1, 1);
            
            % Create the place to store the errors
            this.errors = zeros(this.numCases, this.MAX_ITERATIONS);
        end
        
        % The function that does the actual work
        function train(this)
            % Iterate through however many times (like 10,000, more means
            %   higher precision
            for it=1:1:this.MAX_ITERATIONS
                % Go through every case
                for ca=1:1:this.numCases
                    
                    % FEED FORWARD ----------------------------------------
                    
                    % The initial inputs and 1 for the bias
                    initInputs = [ this.trainingInputs(:,ca) ; 1 ];
                    
                    % The outputs of the hidden layer
                    houtputs = zeros(this.numNeurons+1, 1);
                    
                    % INITIAL INPUTS -> HIDDEN LAYER
                    for ne=1:1:this.numNeurons
                        % ACTIVATION
                        houtputs(ne,:) = LReLU( dot(initInputs, this.hweights(:, ne)) );
                    end
                    % Add in a bias component
                    houtputs(this.numNeurons+1, 1) = 1;
                    
                    % HIDDEN LAYER -> OUTPUT
                    output = Sigma( dot(houtputs, this.oweights) );
                    
                    
                    %Find the error and log it
                    netError = this.idealOutputs(ca, 1) - output;
                    this.errors(ca, it) = netError;
                    
                    
                    % BACK PROPAGATE --------------------------------------
                   
                    % Derivatives for each neuron
                    derivs = houtputs * netError * SigDeriv( dot(houtputs, this.oweights) );
                    
                    % Change the weights for the hidden layer and output
                    
                    % HIDDEN LAYER -> OUTPUT
                    for ne=1:1:this.numNeurons+1
                        % Add lc * deriv for this neuron
                        this.oweights(ne,1) = this.oweights(ne,1) + ( this.learningCoefficient * derivs(ne,1) );
                    end
                    
                    % INITIAL INPUTS -> HIDDEN LAYER
                    for ne=1:1:this.numNeurons
                        
                        % Get the sig deriv for this neuron
                        sd = LReLUD( dot(initInputs, this.hweights(:,ne)) );
                        
                        for in=1:1:this.numInputs+1
                            % For each input variable, subtract lc * deriv
                            %   for the neuron * the sigderiv for the
                            %   neuron
                            this.hweights(in, ne) = this.hweights(in, ne) - (this.learningCoefficient * derivs(ne,:) * initInputs(in,1) * sd);
                        end
                        
                    end
                    
                end
            end
            
            figure();
            i = 1:1:this.MAX_ITERATIONS;
            
            hold on
            for j=1:1:this.numCases
                plot(i, abs(this.errors(j,:)));
            end
            hold off
            
        end
        % END TRAIN
    end
    
end

