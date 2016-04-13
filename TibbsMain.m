clc;
clear;

network = TibbsNetwork(3, 10, 500, 'mux.txt', 0.5);

network.train();