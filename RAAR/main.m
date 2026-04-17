function[labels]=main(X,cls_num,anchor,alpha,gamma,delta,beta)
    warning off
    nV=length(X);  %the number of views
    N=size(X{1},2);%the number of samples
    t=anchor;      %the number of anchors

    for k=1:nV
        X{k}=X{k}';
        A{k}=zeros(t,size(X{k},2));%anchor matrix
        B{k}=zeros(t,size(X{k},2));
        C{k}=zeros(t,size(X{k},2));
        Z{k}=zeros(N,t);%anchor graph 
        D{k}=zeros(N,t);%diversity matrix 
        G{k}=zeros(N,t);
        W{k}=zeros(N,t); 
        F{k}=zeros(N,t);%consistency matrix 
        H{k}=zeros(N,t);
        K{k}=zeros(N,t);
        L{k}=zeros(N,t);
    end

    w=zeros(N*t*nV,1);
    g=zeros(N*t*nV,1);
    sX=[N,t,nV];

    Isconverg=0; epson=1e-7;
    iter=0;
    rho=0.0001; max_rho=10e12; pho_rho=2;%rho相关参数

    while(Isconverg==0)
        %Update Z
        for k=1:nV
            value(k) = norm(X{k}-Z{k}*A{k}, 'fro')^2;
            tmpz=2*A{k}*A{k}'+rho*(delta^2+value(k))*eye(t,t);
            Z{k}=(2*X{k}*A{k}'+rho*(D{k}+F{k}-L{k}/rho)*(delta^2+value(k)))*pinv(tmpz); 
            Z{k}(isnan(Z{k})) = 0;Z{k}(isinf(Z{k})) = 0;
        end
        
        %Update G
        D_tensor=cat(3,D{:,:});
        W_tensor=cat(3,W{:,:});
        d=D_tensor(:);
        w=W_tensor(:);
        [g, ~] = wshrinkObj(d+w/rho,alpha/rho,sX,0,3,gamma);%w*
        w=w+rho*(d-g);
        G_tensor = real(reshape(g, sX));
        W_tensor = real(reshape(w, sX));

        for k=1:nV
            W{k}=W_tensor(:,:,k); 
            G{k}=G_tensor(:,:,k);
        end
    
        %Update A
        for k=1:nV
            tmpa=2*Z{k}'*Z{k}+(rho*(delta^2+value(k))*eye(t,t));
            A{k}=pinv(tmpa)*(2*Z{k}'*X{k}+(rho*B{k}-C{k})*(delta^2+value(k))); 
            A{k}(isnan(A{k})) = 0;A{k}(isinf(A{k})) = 0;
        end       
        
        %Update B
        for k=1:nV
            tempB{k}=A{k}+C{k}/rho;
            [Au,~,Av]=svd(tempB{k},'econ');
            B{k}=Au*Av';
            B{k}(isnan(B{k})) = 0;B{k}(isinf(B{k})) = 0;
        end
        
        %Update D
        for k=1:nV
            D{k}=0.5*(G{k}-W{k}/rho+Z{k}-F{k}+L{k}/rho); 
        end

        %Update F
        for k=1:nV
            F{k}=0.5*(H{k}-K{k}/rho+Z{k}-D{k}+L{k}/rho); 
        end

        %Update H
        for k=1:nV
            H{k} = solve_l1l2new2(F{k}+K{k}/rho, 1/rho,beta);
        end

        %Update Multipliers 
        for k=1:nV
            C{k}=C{k}+rho*(A{k}-B{k});
            K{k}=K{k}+rho*(F{k}-H{k});
            L{k}=L{k}+rho*(Z{k}-D{k}-F{k});
        end
        rho=min(rho*pho_rho,max_rho);
        
        %convergence, 0 yes, 1 no
        Isconverg=1;
        for k=1:nV
            if(norm(X{k}-Z{k}*A{k},inf)>epson)
                history.norm_Z=norm(X{k}-Z{k}*A{k},inf);
                Isconverg=0;
            end

            if(norm(A{k}-B{k},inf)>epson)
                history.norm_A_B=norm(A{k}-B{k},inf);
                Isconverg=0;
            end

            if(norm(F{k}-H{k},inf)>epson)
                history.norm_F_H=norm(F{k}-H{k},inf);
                Isconverg=0;
            end

            if(norm(Z{k}-D{k}-F{k},inf)>epson)
                history.norm_Z_D_F=norm(Z{k}-D{k}-F{k},inf);
                Isconverg=0;
            end
            
            if(norm(D{k}-G{k},inf)>epson)
                history.norm_Z_G=norm(Z{k}-G{k},inf);
                Isconverg=0;
            end
        end

        if(iter>50)
            Isconverg=1;
        end
        iter=iter+1;

    end
    
    %clustering
    Sbar=[];
    for k=1:nV
        Sbar=cat(1,Sbar,1/sqrt(nV)*Z{k}');
    end
    [U,~,~]=mySVD(Sbar',cls_num);
    rand('twister',5489)
    labels=litekmeans(U,cls_num,'MaxIter',100,'Replicates',10);

  
end