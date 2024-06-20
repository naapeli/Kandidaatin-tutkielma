function Reco = compute_reco(X,data,Gamma_noise,invGamma_prior,rounds,T,gamma,N,tol,krylov)

XT = X';
summa = 0;
invGamma_noise = inv(Gamma_noise);
epsilon = sqrt(length(data));

% iterate at most rounds of rounds
for j=1:rounds
    
    invH = @(x) (invGamma_prior\x);
    
    % deduce a solution via prior-conditioning, LSQR and early stopping
    if krylov
        Ln = sqrt(invGamma_noise);
        datatilde = Ln*data;
        Xtilde = Ln*X;
        afun = @(x,t) solhelper(x,t,Xtilde);
        [reco, ~, ~, iter, resvec, ~, ~, ~, Vs, ~] = mlsqr2(afun, datatilde, [], epsilon/norm(datatilde), [], 100, invH, 'left', 0,[], [0 0 0 1]);
    
    % or as standard posterior mean
    else
        B = Gamma_noise + X*invH(XT);
        reco = invH(XT*(B\data));
    end
    
    Reco = reshape(reco,N,N);
    Reunat = gradient(Reco);
    summa_old = summa;
    summa = sum(sum(Reunat)); % an approximate scaled L1-norm of the gradient
    
    abs(summa-summa_old)/summa  % display the relative decrease
    
    % if below tolerance, break
    if abs(summa-summa_old)/summa < tol
        break
    end
    
    % otherwise, build a new prior based on the new reconstruction
    paino = 1./sqrt(T^2 + Reunat.^2);
    invGamma_prior = 1/gamma^2*D_DDDD_weight(N,paino);
   
end



