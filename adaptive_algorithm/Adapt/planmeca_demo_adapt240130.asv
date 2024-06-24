tic
close all
clear

%% NH on January 30, 2024

% A script for finding A-optimal projection angles (kind of) for a certain
% square object with a predefined absorption and ROI under an initial Gaussian
% (zero-mean) prior with inverse of the discrete Laplacian with Dirichlet
% (or Neumann) boundary conditions. The intial prior is updated after each
% reconstruction based on the 'edges' detected in the lates reconstruction
% (a la Calvetti, Somersalo, Arridge, Harhanen, Majander etc.).

% The main reference: "T. Helin, N. Hyvönen, and J.-P. Puska, 
% Edge-promoting adaptive Bayesian experimental design for X-ray imaging, 
% SIAM Journal on Scientific Computing, 44, B506-B530 (2022)".

% The code can also be used to simultaneously optimize the positioning of the
% parallel beam source-detector pair: Just choose the proportion of the real
% sensors (pr) to be less than 1. The algorithm then sets the number real sensors
% to be nr=round(pr*n) and considers np=n-nr+1 positions for the sensor-detector
% pair for each angle. This option naturally makes the algorithm quite a bit slower.

% Two distinct regions are to be defined:

% ROI -- the pixels about which we want to gather information
% bg -- background, i.e. the rest of the unit square

% _THIS CODE DOESN'T SUPPORT OBSTRUCTIONS OR D-OPTIMALITY_

% The ROI and the background have a joint prior.

% The noise model is additive Gaussian with independent components with std
% sigma.


%% Setting the stage

% Choose if you form the whole X-ray matrices for simulation, optimization 
% and reconstruction. This needs to be done if discretization-related
% parameters are changed below
form_xray = false; 

% Make wise choices before you can start -- some options may be SLOW:
visualize_variances = false; % visualize variances after each optimization step?
visualize_gradients = false; % visualize the gradients after each optimization step?
draws_from_posterior = false; % visualize random draws from the current posterior?
plot_dominating_eigen_modes = false; % visualize the eigenvector of the (posterior) covariance for the largest eigenvalue
pauseko = false; % whether there is a pause between the optimization rounds
ld = 30; % maximum number of lagged diffusivity steps after a new projection
tol = 10^(-4); % tolerance for stopping the lagged diffusivity iteration
krylov = false; % use LSQR and Morozov for taking the lagged diffusivity steps (DOES NOT WORK WITHOUT EXTRA AUXILIARY FUNCTIONS)

%% Basic definitions
J = 10; % number of random targets considered

N = 100; % Discretization for reconstruction
Ndata = 111; % Discretization for simulation of data
Nopt = 31; % Discretization for optimization
n = 51; % number of sensors

pr = 0.25; % proportion of real sensors
nr = round(pr*n); % number of real sensors
np = n-nr+1; % number of possible sensor positions
mm = 50; % number of line integrals per sensor (an overkill!)
nn = mm*n; % number of line integrals per projection
m = 100; % number of projection angles

n_dir = 20; % number of sequentially optimized projections is n_dir+1;
sigma = 0.01; % standard deviation of noise 

s = linspace(-0.49,0.49,nn); % distances of line integrals from the line passing through the origin
phi = pi/(2*m):pi/m:pi-pi/(2*m); % the projection angles

y = linspace(-0.5,0.5,N);
[Y1,Y2] = meshgrid(y,y); % spatial grid over the unit square for reconstruction

y = linspace(-0.5, 0.5, Nopt);
[Y1opt, Y2opt] = meshgrid(y,y); % spatial grid over the unit square for optimization

y = linspace(-0.5, 0.5, Ndata);
[Y1data, Y2data] = meshgrid(y,y); % spatial grid over the unit square for optimization

gamma = 10; % gamma^2 multiplies the inverse Laplacian to get the covariance
T = 10^(-6); % for smnoothening the absolute value in the "TV prior"

%% Determine ROI and background

regions = 0;

switch regions
    case 0
        % everything is ROI
        ROI = true(size(Y1opt));
        BG = ~ROI;
        
        ROI_reco = true(size(Y1));
    case 1
        ROI = (abs(Y1opt)<0.25 & abs(Y2opt)<0.25);
        BG = ~ROI;
        
        ROI_reco = (abs(Y1)<0.25 & abs(Y2)<0.25);
end


roi = ROI(:); % indices of the ROI pixels
roi_reco = ROI_reco(:);
Weight = 1.0*diag(roi); % diagonal weight for ROI (for A-optimality)

% Inverse covariances for the prior
invGamma_prior = 1/gamma^2*D_DDDD_weight(N,ones(N));
invGamma_prior_opt = 1/gamma^2*D_DDDD_weight(Nopt,ones(Nopt));

% depending on the size of the problem, this may be reasonable
Gamma_prior_opt = inv(invGamma_prior_opt);
Gamma_prior_opt = full(Gamma_prior_opt);

% store the intial inverse prior
invGamma_prior0 = invGamma_prior; 

% plot the ROI
% figure(999)
% pcolor(Y1opt,Y2opt,real(ROI));
% title('ROI')
% axis square, shading flat
% drawnow

%% The projection matrix including all angles

if form_xray

    display('Forming the X-ray projection matrix')
 
    % The X-ray projection matrix: All combinations of the lines s and angles
    % phi are initially considered. Subsequently, a row for the final
    % projection matrix, i.e., a row corresponding to a single sensor, is
    % obtained by taking the mean of mm consecutive ows. In kokoX, the rows corresponding
    % to a single projection angle are consecutive. In par ular, length(s)
    % must be divisible by mm -- as it is in this script (nn = mm*n).

    % for reconstruction
    X = Xraymat_sum_rows(s, phi, N, mm);
    XT = X';

    % for optimization
    X_opt = Xraymat_sum_rows(s, phi, Nopt, mm);
    XT_opt = X_opt';

    % for data simulation
    X_data = Xraymat_sum_rows(s, phi, Ndata, mm);
    XT_data = X_data'; 

    display('Done')

    save Xray_matriisit X X_opt X_data

else

    load Xray_matriisit

    % The transposes are the ones that are actually manipulated
    XT = X';
    XT_opt = X_opt';
    XT_data = X_data';
end

%% Define the standard deviation of noise
    
    standd = sigma*ones(n*m,1); % a very simple noise model
    var = standd.^2;


%% loop over the random targets

for jjj=1:J

    virhe = 1; % set the relative L^2 error to one
    

    % Define the target absorption
    kohde = 0;
    
    switch kohde
    
        case 0
    
            target = zeros(size(Y1)); % for evaluating reconstruction quality
            target_data = zeros(size(Y1data)); % for data simulation
            ellipses = 1+unidrnd(4); % how many ellipses in a target
            
            for j=1:ellipses
    
                % the main axes of the considered ellipse
                R1 = 0.15*(rand(1)+1/3);
                R2 = 0.15*(rand(1)+1/3);
                theta = 2*pi*rand; % angle of the ellipse center w.r.t. the origin
                r = 0.5*sqrt(rand); % distance of the ellipse center from the origin
                c = r*[cos(theta); sin(theta)]; % the ellipse center
                kulma = 2*pi*rand(1); % orientation of the ellipse
                % corresponding rotation matrix
                A = [cos(kulma), -sin(kulma); sin(kulma), cos(kulma)];
                level = 0.5+rand; % random absorption level of the ellipdes
    
                % assigning "level" absorption value for all pixels inside the
                % ellipse (unreadable -- sorry)
                target = target + level.*( (A(1,1)*Y1+A(1,2)*Y2-c(1)).^2/R1^2+(A(2,1)*Y1+A(2,2)*Y2-c(2)).^2/R2^2 < 1 );
                target_data = target_data + level.*( (A(1,1)*Y1data+A(1,2)*Y2data-c(1)).^2/R1^2+(A(2,1)*Y1data+A(2,2)*Y2data-c(2)).^2/R2^2 < 1 );
            end 
                
    end
    
    % plot the target
    figure('position',[0, 250, 550, 550])
    pcolor(Y1,Y2,target),
    axis square, shading flat, colorbar
    caxis([min(target(:)),max(target(:))])
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    title('The target')
    drawnow
    
    
    % Example draws from the prior
    
    if draws_from_posterior
        figure
        % Cholesky factor of the inverse prior
        Ch = chol(invGamma_prior);
        for j=1:4
            white_noise = randn(N^2,1);
            sample = Ch\white_noise;
            proj = X*sample;
            Sample = reshape(sample,N,N);
            subplot(2,2,j);
            imagesc(Sample)
            % the largest value in the projections corresponding to the randomly
            % drawn image
            title(num2str(max(abs(proj))));
            axis square, colorbar
        end
    
        drawnow
    end
    
    
    % introducing different storage matrices and vectors
    AOptimi = []; 
    AInd = [];
    data = [];
    tempXstoreT = [];
    tempXstoreT_opt = [];
    temp_var_store = [];
    Gamma_opt = Gamma_prior_opt; % the previous prior is the initial prior to begin with
    

    % loop over the number of considered projections
    for k=1:n_dir
        Aopt = [];
    
        Gamma_w = Gamma_opt*Weight; % A-weighted prior, i.e., posterior from previous step
    
        str = ['Computing the ' num2str(k) 'th optimality target as a function of the angle'];
        display(str)
    
         
        % loops over projection angles and lateral positions
        for j=1:m
            for jj=1:np
    
                % see "M. Burger, A. Hauptmann, T. Helin, N. Hyvönen, and J.-P. Puska, 
                % Sequentially optimized projections in X-ray imaging, 
                % Inverse Problems, 37, 075006 (2021)"
    
                % the rows corresponding to the jth projection angle and jjth
                % position, accounting only for nonobstructed detectors
                tempXT = XT_opt(:,(j-1)*n+jj:j*n-np+jj); % the potentially acceptable rows
                tempX = tempXT';
    
                temp_var = var((j-1)*n+jj:j*n-np+jj);
                Gamma_noise = diag(temp_var);
    
                % Matrix that needs to be inverted to get the posterior covariance.
                % These are not necessarily optimal expressions for all matrix sizes. 
                B = Gamma_noise + tempX*Gamma_opt*tempXT;
                C = chol(B);
                Apu = C'\(tempX*Gamma_w); 
    
                % weighted trace of the posterior, without caring about the
                % baseline
                aopt = - sum(sum(Apu.^2));
                
                Aopt = [Aopt, aopt]; % save the result
                
            end
    
    
            % monitor the progression
            if mod(j,10)==0
                display([k,n_dir,j,m])
            end
    
        end
    
         
        display('Done')
    
        [aoptimi,aind] = min(Aopt);  % optimal value and its index
        indj = ceil(aind/np); % index of the optimal projection angle
        indjj = mod(aind-1,np)+1; % index of the optimal lateral position
    
        % saving and displaying stuff
        AOptimi = [AOptimi aoptimi];
        AInd = [AInd aind];
        display([AOptimi; 180/pi*phi(ceil(AInd/n))])
    
        % projection matrix for reconstruction (thus far)
        tempXT = XT(:,(indj-1)*n+indjj:indj*n-np+indjj); % it is faster to manipulate columns
        tempX = tempXT'; % the new projections
        tempXstoreT = [tempXstoreT, tempXT]; % all projections thusfar
        tempXstore = tempXstoreT';
    
        % same for the projection matrix for the optimization
        tempXT_opt = XT_opt(:,(indj-1)*n+indjj:indj*n-np+indjj); % it is faster to manipulate columns
        tempX_opt = tempXT_opt'; % the new projections
        tempXstoreT_opt = [tempXstoreT_opt, tempXT_opt]; % all projections thusfar
        tempXstore_opt = tempXstoreT_opt';
    
        % (almost) same for data simulation
        tempXT_data = XT_data(:,(indj-1)*n+indjj:indj*n-np+indjj); % it is faster to manipulate columns
        tempX_data = tempXT_data'; % the new projections
        
        % noise and other stuff
        temp_var = var((indj-1)*n+indjj:indj*n-np+indjj); % the current noise variance
        temp_var_store = [temp_var_store; temp_var]; % all noise variances thusfar
        Gamma_noise_store = diag(temp_var_store); % the noise covariance thus far
        invGamma_noise_store = diag(1./temp_var_store);
            
        % generating new noisy data
        data_new = tempX_data*target_data(:);
        data_new = data_new + sqrt(temp_var).*randn(size(data_new));
        data = [data; data_new]; % all noisy data thus far
    
        % computing the reconstructions
        Reco = compute_reco(tempXstore,data,Gamma_noise_store,invGamma_prior,ld,T,gamma,N,tol,krylov);
        virhe = [virhe, sqrt(sum((target(roi_reco)-Reco(roi_reco)).^2))/sqrt(sum(target(roi_reco).^2))];
    
        % 'detecting' the edges
        Reunat = gradient(Reco); % the absolute value of the gradient for Reco
        % the same for the optimization resolution
        Reunat_opt = interp2(Y1,Y2, Reunat, Y1opt, Y2opt);
    

        % Visualizing the newest reconstruction
        figure('Position',  [840, 250, 550, 550])
        pcolor(Y1, Y2, Reco)
        axis square, shading flat, colorbar
        caxis([min(min(target)),max(max(target))])
        hold on
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])

        % position of the parallel beam device
        t = linspace(-1,1,2);
        ss1 = s((indjj-1)*mm+1);
        x1 = ss1*cos(phi(indj)) - t*sin(phi(indj));
        y1 = -(ss1*sin(phi(indj)) + t*cos(phi(indj)));
        plot(x1,y1, 'k', 'Linewidth', 3)

        ss2 = s((indjj-1+nr)*mm);
        x2 = ss2*cos(phi(indj)) - t*sin(phi(indj));
        y2 = -(ss2*sin(phi(indj)) + t*cos(phi(indj)));
        plot(x2,y2, 'k', 'Linewidth', 3)
        
        
        if visualize_gradients
            % Visualizing the 'edges' in the reconstruction
            figure
            pcolor(Y1, Y2, Reunat)
            axis square, shading flat, colorbar
            hold on

            t = linspace(-1,1,2);
            ss1 = s((indjj-1)*mm+1);
            x1 = ss1*cos(phi(indj)) - t*sin(phi(indj));
            y1 = -(ss1*sin(phi(indj)) + t*cos(phi(indj)));
            plot(x1,y1, 'k', 'Linewidth', 3)

            ss2 = s((indjj-1+nr)*mm);
            x2 = ss2*cos(phi(indj)) - t*sin(phi(indj));
            y2 = -(ss2*sin(phi(indj)) + t*cos(phi(indj)));
            plot(x2,y2, 'k', 'Linewidth', 3)

            title(['Absolute value of gradient for the reconstruction after ' num2str(k) ' projections'])
        end
    
        % Compute a 'TV weight' based on the 'edges' in the reconstruction
        paino = 1./sqrt(T^2 + Reunat.^2);
        paino_opt = 1./sqrt(T^2 + Reunat_opt.^2);
    
        % computing new _initial_ (inverse) prior based on the (edges in the) reconstruction
        invGamma_prior = 1/gamma^2*D_DDDD_weight(N,paino);
        invGamma_prior_opt = 1/gamma^2*D_DDDD_weight(Nopt,paino_opt);
        
        % the posterior for the optimization process
        invGamma_opt = invGamma_prior_opt + tempXstoreT_opt*invGamma_noise_store*tempXstore_opt;
        Gamma_opt = inv(invGamma_opt);
    
        % Plotting eigenvectors
        if plot_dominating_eigen_modes
            
            % eigenvector corresponding to the largest eigenvalue of the prior 
            % i.e. the smallest one of the inverse prior
            [Vprior Dprior Uprior] = svds(invGamma_prior_opt, 1,  'smallest');
        
            % taking also all measurements thus far into account, i.e. the
            % posterior, i.e. the prior for the next optimization round
            [Vpost Dpost Upost] = svds(invGamma_opt, 1, 'smallest');
        
            figure
            pcolor(reshape(Vprior,Nopt,Nopt)), axis square, shading flat
            title('Largest prior eigenvector')
            figure
            pcolor(reshape(Vpost,Nopt,Nopt)), axis square, shading flat
            title('Largest posterior eigenvector')
        end
        
        if draws_from_posterior
            figure
            
            invGamma = invGamma_prior + tempXstoreT*invGamma_noise_store*tempXstore;
            CCC = chol(invGamma);

            for j=1:4
                white_noise = randn(N^2,1);
                sample = CCC\white_noise;
                proj = X*sample;
                Sample = reshape(sample,N,N);
                subplot(2,2,j);
                pcolor(Y1,Y2,Sample+Reco), shading flat
                % the largest value in the projections corresponding to the randomly
                % drawn image
                title(num2str(max(abs(proj))));
                axis square, colorbar
            end
        end

        if visualize_variances
            figure
            % This isn't a reasonable implementation
            invGamma = invGamma_prior + tempXstoreT*invGamma_noise_store*tempXstore;
            Gamma = inv(invGamma);
            apu = diag(Gamma);
            KokoVari = reshape(apu,N,N);
            pcolor(Y1,Y2,sqrt(KokoVari))
            hold on

            t = linspace(-1,1,2);
            ss1 = s((indjj-1)*mm+1);
            x1 = ss1*cos(phi(indj)) - t*sin(phi(indj));
            y1 = -(ss1*sin(phi(indj)) + t*cos(phi(indj)));
            plot(x1,y1, 'k', 'Linewidth', 3)

            ss2 = s((indjj-1+nr)*mm);
            x2 = ss2*cos(phi(indj)) - t*sin(phi(indj));
            y2 = -(ss2*sin(phi(indj)) + t*cos(phi(indj)));
            plot(x2,y2, 'k', 'Linewidth', 3)

            axis square, colorbar, shading flat
            title(['STD after ' num2str(k) ' projections'])
            
            figure  
            % This isn't a reasonable implementation
            Gamma_prior = inv(invGamma_prior);
            apu = diag(Gamma_prior);
            KokoVari = reshape(apu,N,N);
            pcolor(Y1,Y2,sqrt(KokoVari))
            hold on

            t = linspace(-1,1,2);
            ss1 = s((indjj-1)*mm+1);
            x1 = ss1*cos(phi(indj)) - t*sin(phi(indj));
            y1 = -(ss1*sin(phi(indj)) + t*cos(phi(indj)));
            plot(x1,y1, 'k', 'Linewidth', 3)

            ss2 = s((indjj-1+nr)*mm);
            x2 = ss2*cos(phi(indj)) - t*sin(phi(indj));
            y2 = -(ss2*sin(phi(indj)) + t*cos(phi(indj)));
            plot(x2,y2, 'k', 'Linewidth', 3)

            axis square, colorbar, shading flat
            title(['Final prior STD for the ' num2str(k) ' measurement (A)'])
        end
    
    
        drawnow
    
        if pauseko
            display('Hit a key to continue')
            pause
        end
    
    end
    
    
    %% Reference solutions for equiangular projections
    
    n_refdir = round(pr*n_dir);
    virhe_ref = 1;
    
    for j=1:n_refdir
        
            step_angle = m/j;
            
            XrefT = sparse(N^2,j*n);
            Xref_dataT = sparse(Ndata^2,j*n);   
            ref_var = zeros(j*n,1);
            
            for l=1:j
                angle_ind = round(1+(l-1)*step_angle);
                XrefT(:,(l-1)*n+1:l*n) = XT(:,(angle_ind-1)*n+1:angle_ind*n);
                Xref_dataT(:,(l-1)*n+1:l*n) = XT_data(:,(angle_ind-1)*n+1:angle_ind*n);
                ref_var((l-1)*n+1:l*n) = var((angle_ind-1)*n+1:angle_ind*n);
            end
    
            Xref = XrefT';
            Xref_data = Xref_dataT';
            
            data_ref = Xref_data*target_data(:) + sqrt(ref_var).*randn(size(ref_var));
            Gamma_noise_ref = diag(ref_var);
            
            Reco_ref = compute_reco(Xref,data_ref,Gamma_noise_ref,invGamma_prior0,ld,T,gamma,N,tol,krylov);
            virhe_ref = [virhe_ref, sqrt(sum((target(roi_reco)-Reco_ref(roi_reco)).^2))/sqrt(sum(target(roi_reco).^2))]; 
            
            figure('Position',  [840, 250, 550, 550])
            pcolor(Y1, Y2, Reco_ref)
            axis square, shading flat, colorbar
            caxis([min(min(target)),max(max(target))])
            set(gca,'xtick',[])
            set(gca,'xticklabel',[])
            set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            title(['Reference reconstuction with ', num2str(j), ' equiangular full projections'])
            hold on
            
            drawnow
    end
    
    
    %% Visualizing the mean reconstruction error and the related STD thus far
    
    close all
    
    if jjj==1
        mean_virhe = virhe;
        mean_virhe_ref = virhe_ref;
        var_virhe = zeros(size(virhe));
        var_virhe_ref = zeros(size(virhe_ref));
        std_virhe = sqrt(var_virhe);
        std_virhe_ref = sqrt(var_virhe_ref);
    else
        mean_virhe_old = mean_virhe;
        mean_virhe = ((jjj-1)*mean_virhe + virhe)/jjj;
        
        mean_virhe_ref_old = mean_virhe_ref;
        mean_virhe_ref = ((jjj-1)*mean_virhe_ref + virhe_ref)/jjj;
        
        var_virhe = (jjj-1)*var_virhe + (virhe-mean_virhe).*(virhe-mean_virhe_old);
        var_virhe = var_virhe/jjj;
        std_virhe = sqrt(var_virhe);
        
        var_virhe_ref = (jjj-1)*var_virhe_ref + (virhe_ref-mean_virhe_ref).*(virhe_ref-mean_virhe_ref_old);
        var_virhe_ref = var_virhe_ref/jjj;
        std_virhe_ref = sqrt(var_virhe_ref);
    end
    
    figure(666)
    hold on
    errorbar(0:length(mean_virhe)-1,mean_virhe, std_virhe,'b-*')
    errorbar([0,round(1/pr*(1:n_refdir))],mean_virhe_ref, std_virhe_ref,'r-d')
    title(['Error at ', num2str(jjj)])
    hold off

end

toc