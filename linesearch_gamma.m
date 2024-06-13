function gamma_new = linesearch_gamma(Y, TempX, Temp_var, proibg, pnonroi, clength, roi, nonroi, bg, n_line, lower, upper)

n = size(Y,2); % the number of projections thus far
gr = (1 + sqrt(5))/2; % golden ration

% the prior covariance with standard deviation 1
Gamma_basic = form_prior(proibg, pnonroi, [1,1], clength, roi, nonroi, bg);

% initialization
gamma1 = [upper - (upper-lower)/gr, 1]; % lower intermediate point
gamma2 = [lower + (upper-lower)/gr, 1]; % higher intermediate point
Phi1 = 0;
Phi2 = 0;

% Compute the values of the target function at gamma1 and gamma2.
% See formula \eqref{eq:likelihood} in the paper.
for k=1:n
    Upsilon1 = diag(Temp_var{k}) + gamma1(1)^2*TempX{k}*Gamma_basic*TempX{k}';
    Ch1 = chol(Upsilon1);
    apu1 = Ch1'\Y{k};
    Phi1 = Phi1 + 2*sum(log(diag(Ch1))) + sum(apu1.^2);
            
    Upsilon2 = diag(Temp_var{k}) + gamma2(1)^2*TempX{k}*Gamma_basic*TempX{k}';
    Ch2 = chol(Upsilon2);
    apu2 = Ch2'\Y{k};
    Phi2 = Phi2 + 2*sum(log(diag(Ch2))) + sum(apu2.^2);
end

% golden section line search with n_line divisions
for j=1:n_line
     
    % display the current search interval 
    display([lower, upper])
    % the target values at the current intermediate points
    display([Phi1, Phi2])
    
    % If Phi1 < Phi2, then new upper bound is gamma2, new gamma2 is the
    % former gamma1, and the value of the target at new gamma1 needs to be
    % computed.
    if Phi1 < Phi2
        upper = gamma2(1);
        Phi2 = Phi1;
        gamma2 = gamma1;
        gamma1 = [upper - (upper-lower)/gr, 1];
        
        % compute the target value at the new gamma1
        for k=1:n
            Phi1 = 0;
            Upsilon1 = diag(Temp_var{k}) + gamma1(1)^2*TempX{k}*Gamma_basic*TempX{k}';
            Ch1 = chol(Upsilon1);
            apu1 = Ch1'\Y{k};
            Phi1 = Phi1 + 2*sum(log(diag(Ch1))) + sum(apu1.^2);
        end
    
    % If Phi1 > Phi2, then new lower bound is gamma1, new gamma1 is the
    % former gamma2, and the value of the target at new gamma2 needs to be
    % computed.   
    else
        lower = gamma1(1);
        Phi1 = Phi2;
        gamma1 = gamma2;
        gamma2 = [lower + (upper-lower)/gr, 1];
        %Gamma2 = form_prior(proibg, pnonroi, gamma, clength2, roi, nonroi, bg);
        
        % compute the target value at the new gamma2
        for k=1:n
            Phi2 = 0;
            Upsilon2 = diag(Temp_var{k}) + gamma2(1)^2*TempX{k}*Gamma_basic*TempX{k}';
            Ch2 = chol(Upsilon2);
            apu2 = Ch2'\Y{k};
            Phi2 = Phi2 + 2*sum(log(diag(Ch2))) + sum(apu2.^2);
        end
        
    end
   
    
end

% choose between the current gamma1 and gamma2
if Phi1 < Phi2
    gamma_new = gamma1;
else
    gamma_new = gamma2;
end

