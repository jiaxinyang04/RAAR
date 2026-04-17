clear;
folder_now = pwd;
addpath([folder_now, '\funs']);
addpath([folder_now, '\dataset']);
%% ==================== Load Datatset and Normalization ===================
load("CiteSeer.mat");
X=fea;
cls_num=length(unique(gt));
for v=1:length(X)
    [X{v}]=NormalizeData(X{v}');
end
%% ============================ Optimization ==============================
anc =[cls_num,2*cls_num,3*cls_num,4*cls_num,5*cls_num,6*cls_num,7*cls_num,8*cls_num];
gamma=[1,2];
i=-2; j=0; k=7; beta=0.5;
alpha = 10^(i); delta = 10^(j); anchor = anc(k);
y = main(X, cls_num, anchor,alpha,gamma,delta,beta);
result=  ClusteringMeasure(gt, y);% [ACC MIhat Purity F P R AR];
fprintf('%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\n',...\
        ' ACC','NMI','PUR','F-score','PRE','Recall','AR');
fprintf('\n%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
         result(1),result(2),result(3),result(4), result(5),result(6),result(7)); 
         
              
             

