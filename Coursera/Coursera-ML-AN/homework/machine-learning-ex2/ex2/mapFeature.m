function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

degree = 6;
%size(x(:,1))返回的是x第一列的行数和列数，显然列数为1
%ones(n) n*n矩阵
out = ones(size(X1(:,1)));
%特征映射后特征变多，相应的数据项也应该变多，这些新的实例由旧的实例中来
%当end作为下标时，代表最后一个可用元素，比如说a=[1 2; 3 4; 5 6]那么a(1,end)=2代表第一行最右边一个元素，a(end, 1)=5代表最后一行第一个元素,所以下面的例子相当于每次都插进新的一行，也意味着第一行就是纯1.

for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end
