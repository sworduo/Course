function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

J = 1/2*sum(sum(((X*Theta'-Y).^2).*R))+lambda/2*sum(sum(Theta.^2))+lambda/2*sum(sum(X.^2));

%对于梯度的每一项X_grad(i, k)计算公式的理解：
%计算所有用户对第i行电影的预测值，减去实际的y值得到差，
%然后点乘theta矩阵的第k列转置，相当于点乘每个用户对该点x的导数值
%每个用户的公式：θ1X1+θ2X2+θ3X3+.。。
%对某部电影的第k个特征求导就是每个用户的第k个参数。
%第二个循环也是类似的，不同的是每一次按列向量计算，即求每个用户对所有电影的得分。
%注意这里点乘R是因为不是每个用户都对电影评分了。
%for i= 1:num_movies
%	for k= 1:num_features
%	X_grad(i, k) = sum((X(i, :)*Theta'-Y(i, :)).*Theta(:, k)'.*R(i, :));
%	end;
%end;

%for j = 1:num_users
%	for k = 1:num_features
%		Theta_grad(j, k) = sum((X*Theta(j, :)' - Y(:, j)).*X(:, k).*R(:, j));
%	end;
%end;

%向量形式
%事实上，预测值与实际值的差是一个行向量，
%第i部电影第一个特征需要这个向量与所有用户第一个θ点乘的和
%第二个特征需要这个向量与所有用户第二个θ点乘的和
%其实只要用这个向量与Theta直接相乘即可！！！
for i =1:num_movies
	idx = find(R(i, :)==1);
	Ttmp = Theta(idx, :);
	Ytmp = Y(i, idx);
	X_grad(i, :) = (X(i, :)*Ttmp'-Ytmp)*Ttmp+lambda*X(i, :);
end;

for j = 1:num_users
	idx = find(R(:, j)==1);
	Xtmp = X(idx, :);
	Ytmp = Y(idx, j);
	Theta_grad(j, :) = (Xtmp'*(Xtmp*Theta(j, :)'-Ytmp))'+lambda*Theta(j, :);
end;










% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
