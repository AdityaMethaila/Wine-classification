%starter code for project 2: linear classification
%pattern recognition, CSE583/EE552
%Weina Ge, Aug 2008

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load the WINE data
close all;
clear all;
wine = importdata('./data/wine.data');
%get the class label for each sample
winelabel = wine(:,1);
%each row of winefeature matrix contains the 13 features of each sample
winefeature = wine(:,2:end);

%generate training and test data
train = []; %stores the training samples
test = [];  %stores the test samples
for i=1:3
    ind{i} = find(winelabel==i);
    len = length(ind{i});
    t = randperm(len);
    half = round(len/2);
    train = [train; wine(ind{i}(t(1:half)), :)];
    test = [test; wine(ind{i}(t(half+1:end)), :)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load the FACE data
TrainMat=dlmread('./data/TrainMat.txt');
TestMat=dlmread('./data/TestMat.txt');
LabelTrain=dlmread('./data/LabelTrain.txt');
LabelTest=dlmread('./data/LabelTest.txt');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
