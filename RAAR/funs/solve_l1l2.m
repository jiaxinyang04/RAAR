function [E] = solve_l1l2(W,lambda)%slove_l1l2里面的参数第一个是F范数减去的那个东西，第二个是21范数的系数
n = size(W,2);%样本个数
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);%每一列单独更新
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end