function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
%find函数，返回y==1的序号（octave里的序号是先列后行），因为y只有一列，所以就相当于返回y=1的行数。
pos=find(y==1);
neg=find(y==0);
plot(X(pos, 1), X(pos, 2), 'k+', 'MarkerSize', 7, 'LineWidth', 2);
%MarkerFaceColor代表圆圈的颜色是黄色，k代表标记，也就是这里圆圈的外框是黑色
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);






% =========================================================================



hold off;

end
