function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
%参数传进来的时候只能传一个列向量，所以需将θ矩阵向量化并重新展开为矩阵
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%forward p
X=[ones(size(X,1), 1), X];
%a1是一个矩阵，行对应hid层的节点，列代表每个节点的每个实例，hid_size*m矩阵
a2=sigmoid(Theta1*X');
%a2和X不同，a2行代表特征，列代表实例
a2=[ones(1, size(a2,2)); a2];
a3=sigmoid(Theta2*a2);
%转置后a3行代表实例
%此时a3行代表第几个实例，行代表实例，列代表第几个输出单元的输出
a3=a3';
a2 = a2';

%tmp记录每一个分类的所有实例的矩阵
%一个分类的所有实例应该是m个num_labels维的向量
tmp = zeros(size(a3));

%一个实例应该对应一组向量，但是y只有一个值，所以要想办法把y拓展成相应输出向量
%错误原因在于，每个y所对应的输出向量应该只计算一次，最后得到一个m*num_labels的矩阵逐个相加。
%而下面的做法，则是把一个y值展开成了10个向量，计算了十次，得到了十个m*num_labels矩阵，所以结果偏大了许多
%/*错误做法
%for i=1:num_labels
%	y1 = (y==i);
%	ytmp = zeros(length(y), num_labels);
%	ytmp(:, i) = y1;
%	tmp = tmp-ytmp.*log(a2)-(1-ytmp).*log(1-a2);
%	end;
%*/

%a3是输出层的输出，tmp是代价函数记录每个实例代价的矩阵
for i=1:length(y)
	y1 = zeros(1, num_labels);
	y1(y(i))=1;
	tmp(i, :) = -y1.*log(a3(i, :))-(1-y1).*log(1-a3(i,:));
end;


J = 1/(m)*sum(sum(tmp));

%计算正则项Theta
reg = sum(sum([Theta1(:, 2:end)].^2))+sum(sum([Theta2(:, 2:end)].^2));

J = J + lambda/(2*m)*reg;


delta3 = zeros(num_labels, 1);
delta2 = zeros(hidden_layer_size, 1);

for i = 1:m
	y1 = zeros(num_labels, 1);
	y1(y(i)) = 1;
	%a2a3行代表实例
	delta3 = a3(i, :)' - y1;
	%X中不包括第二层δ0的输入
	delta2 = Theta2'*delta3.*[0;sigmoidGradient(Theta1*X(i, :)')];
	%去掉第二层的δ0是因为，δ0是新增的，与第一层的特征毫无关系，J对第一层θ求导的连式法则不经过δ0，所以需要删掉。
	delta2 = delta2(2:end);
	%本身a2 X的行就代表了实例，也即是a0 a1 a2这些，所以下面不需要转置
	Theta2_grad = Theta2_grad + delta3*a2(i, :);
	Theta1_grad = Theta1_grad + delta2*X(i, :);
end;
	Theta2_grad = 1/(m)*Theta2_grad;
	Theta1_grad = 1/(m)*Theta1_grad;

	%梯度计算正则项
	%注意每一层的theta0都不需要正则化
	reg1 = [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)*lambda/m];
	Theta1_grad = Theta1_grad + reg1;
	reg2 = [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)*lambda/m];
	Theta2_grad = Theta2_grad +reg2;














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%a(:)将矩阵a以列向量的形式将整个矩阵打印出来，组合成一个新的列向量，顺序是，自上而下是第一列、第二列，以此类推。
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
