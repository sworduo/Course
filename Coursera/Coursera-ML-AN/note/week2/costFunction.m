function J = costFunction(x, y ,theta)

	%X 是特征变量，包括x0=2
	%y 是目标变量

	m = size(x,1); %x的行数代表训练集m的大小

	predictions = x*theta;  %常数项x0即θ0已经包含在x里面了

	sqrError = (predictions-y).^2;

	J = 1/(2*m) * sum(sqrError);
