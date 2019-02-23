function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

mu = mean(X);
sigma = std(X);
X_norm = (X - repmat(mu, size(X,1), 1)) ./ repmat(sigma, size(X,1), 1);
%也可以：x_norm=(x-repmat(mu, size(x,1),1)) ./ repmat(sigma,size(x,1),1);
% repmat(a,m,n)代表构建一个m行矩阵，每一行重复a n次，或者说构建一个m×n矩阵，里面每个元素都是a，如果a=[2 2] 那么repmat(a, 3 ,1) 2 2 ; 2 2; 2 2 如果repmat(a, 3, 2)则2 2 2 2; 2 2 2 2; 2 2 2 2(每行两个a)






% ============================================================

end
