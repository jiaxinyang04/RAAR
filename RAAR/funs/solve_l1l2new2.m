function E = solve_l1l2new2(W, lambda,beta)
% 自适应重加权 l21：只输入 W, lambda
% 其它参数写死：iters=5, eps=1e-6
iters = 5; eps_ = 1e-6; 

m = size(W,2);
E = zeros(size(W));
w = ones(1,m);  % 列权重

for t = 1:iters
    % 近端：列 soft-thresholding，阈值 = lambda * w_i
    for i = 1:m
        normi = norm(W(:,i));
        tau = lambda * w(i);
        if normi > tau
            E(:,i) = (1 - tau/normi) * W(:,i);
        else
            E(:,i) = 0;
        end
    end
    % 重加权（IRL21/非凸化）
    coln = sqrt(sum(E.^2,1));               % ||E_:,i||2
    w = 1 ./ ( (coln + eps_).^(1 - beta) );
end
end