% Code for Butterworth filter design

%% Load data
M = csvread('sep1.csv');

n = 2;      % order lowpass digital Butterworth filter
fc = 8;     % cutoff frequency
fs = 100;   % sampled frequency
[b,a] = butter(n, fc/(fs/2));
freqz(b,a);

dataIn = M(:,2);
dataOut = filter(b,a,dataIn);

%% Draw the results
figure(2);
plot(dataIn); hold on;
plot(dataOut);