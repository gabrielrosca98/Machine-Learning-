function [Xtr, Xte, Ytr, Yte]=PartitionData(data, labels, Ntr_per_class)

% Randomly divides the suppiled data into the training and testing sets.
% The training set contains N examples per class. The testing set contains
% all the remaining examples.
%
% PartitionData(data, labels, M)
%
% Arguments: 
%   'data' should be a matrix of n examples (n rows) and d dimensions (d columns).
%   'labels' should be a vector of n examples.
%   'N' should be an integer specifying how many training examples are included per class.
% 
%Returns: Nothing.
% 
% 'N' MUST be an integer less than 10 as there are in total 10 examples for
% each class in the supplied face data.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  YOU SHOULD NOT BE EDITING THIS CODE
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   Xtr =[];
   Xte =[];
   Ytr =[];
   Yte =[];

  
   c = unique(labels);
   for ii = 1:length(c)
      ind = find(labels==c(ii));
      I = randperm(length(ind));
      ind_tr = ind(I(1:Ntr_per_class));
      ind_te = ind(I(Ntr_per_class+1:end));
      
      Xtr = [Xtr; data(ind_tr,:)];
      Xte = [Xte; data(ind_te,:)];
      Ytr = [Ytr; labels(ind_tr)];
      Yte = [Yte; labels(ind_te)];
   end
   
end