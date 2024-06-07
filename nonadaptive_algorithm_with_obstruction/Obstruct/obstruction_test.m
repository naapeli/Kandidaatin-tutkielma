%% NH on January 29, 2024

% Second numerical test

% A script for finding D and or A-optimal projection angles for a square object
% with a predefined ROI under a Gaussian (zero-mean) prior with exponential correlation
% structure (see GaussianDistanceCovariance.m). This script should be able to go at least up 
% to 150 line integrals per 50 sensors and a 100x100 grid on the unit square for D-optimality.
% A-optimality is more tricky and slower. (For D-optimality, the visualization
% of variances is the first thing that starts to break down if the grid
% becomes too dense.)

% The code can also be used to simultaneously optimize the positioning of the
% parallel beam source-detector pair: Just choose the proportion of the real 
% sensors (pr) to be less than 1. The algorithm then sets the number real sensors
% to be nr=round(pr*n) and considers np=n-nr+1 positions for the sensor-detector 
% pair for each angle. This option naturally makes the algorithm quite a bit slower.

% Three distinct regions are to be defined:

% ROI -- the pixels about which we want to gather information
% OBST -- the pixels that cannot be penetrated by X-ray (e.g., due to high absorption)
% bg -- background, i.e. the rest of the unit square

% These sets of pixels can either be a priori mutually independent (having
% their own Gaussian smoothness priors) or alternatively ROI and background
% can have a joint Gaussian smoothness prior (set the variable "roi_bg_correl" accordingly).

% The noise model is additive Gaussian with independent components with std
% sigma (not realistic, I assume).

% Note that the result seems to somewhat depend on the orientation of the pixel
% grid, i.e., on some properties of the discretization. This effect is more pronounced
% if the discretization of the unit square is sparse compared to the number sensors or
% there are only few line integrals per sensor. Altogether, this phenomenon should be
% better understood.

close all
clear all

%% Setting the stage

% Choose if you form the whole X-ray matrices for simulation, optimization 
% and reconstruction. This needs to be done if discretization-related
% parameters are changed below
form_xray = false; 

% Make wise choices before you can start:
visualize_variances = true; % visualize variances after each optimization step?
draws_from_posterior = false; % visualize random draws from the current posterior?
D_optimality = false; % compute D-optimal angles and positions?
A_optimality = true; % compute A-optimal angles and positions?
pauseko = false; % wheather there is a pause between the optimization rounds 

%% Basic definitions

% At the end, use, e.g., these values. Will take some time...
%N = 100; % pixels per edge (the domain is a square)
%n = 45; % total number of virtual sensors

% For testing
N = 49;
n = 21;

pr = 0.33; % proportion of real sensors for
nr = round(pr*n); % number of real sensors
np = n-nr+1; % number of possible sensor positions
mm = 133; % number of line integrals per sensor
nn = mm*n; % number of line integrals per projection
m = 100; % number of projection angles

n_dir = 10; % number of sequentially optimized directions is n_dir
sigma = 0.01; % standard deviation of noise
roi_bg_correl = true; % choose whether roi and background have a common prior

s = linspace(-0.49,0.49,nn); % distances of line integrals from the line passing through the origin
phi = pi/(2*m):pi/m:pi-pi/(2*m); % the projection angles

% form the prior covariance
y = linspace(-0.5,0.5,N); % 
[Y1,Y2] = meshgrid(y,y); % spatial grid over the unit square

% The last components of the following vectors are ignored "if
% roi_bg_correl". The values for OBST are a relic from a previous version,
% where the absorption of the obstruction was high. Here the obstruction
% is non-penetrable.
gamma = [1 1 1]; % pointwise standard deviation of the prior in different regions: ROI, OBST, bg
clength = [0.1 10 0.1]; % correlation length in different regions: ROI, OBST, bg
expt = [1 0 1]; % expectations for the constant absorption in different regions: ROI, OBST, bg


%% Determine ROI, OBST and background, as well as their prior covariances

% Choose the geometry
regions = 2;

switch regions
    case 0 
        ROI = true(size(Y1));
        OBST = ~ROI;
        BG = (~ROI & ~OBST);
    case 1
        ROI = ((Y1-0.1).^2 + (Y2-0.1).^2 < 0.25^2);  % ROI is a disk
        BG = ~ROI; % rest is background
        OBST = (~ROI & ~BG); % no nonROI
    case 2
        OBST = ((abs(Y2) < 0.05) & Y1 < 0);
        ROI = ~OBST;
        BG = (~ROI & ~OBST);
    case 3
        OBST = ((abs(Y2) < 0.05) & Y1 < 0);
        ROI = (~OBST & ((Y1-0.2).^2 + (Y2-0.2).^2 < 0.25^2));
        BG = (~ROI & ~OBST);
end

roi = ROI(:); % indices of the ROI pixels
NNroi = sum(roi); % number of ROI pixels
obst = OBST(:); % indices of the OBST pixels
NNobst = sum(obst); % number of OBST pixels 
bg = BG(:); % indices of the background
NNbg = sum(bg); % number of background pixels 
Weight = diag(roi); % diagonal weight for ROI (for A-optimality)

% Covariances for the prior; small diagonal matrix is added to ensure positive
% definiteness. (There should be a work-around for ensuring
% positive-definiteness in this way.)

if roi_bg_correl % ROI and bg have a common prior
    proibg = [Y1(roi | bg),Y2(roi | bg)];
    Gamma_prior_roi_bg = GaussianDistanceCovariance(proibg,gamma(1),clength(1)) + 0.0001*gamma(1)*eye(NNroi+NNbg);
else % ROI and bg have independent priors
    proi = [Y1(roi),Y2(roi)];
    Gamma_prior_roi = GaussianDistanceCovariance(proi,gamma(1),clength(1)) + 0.0001*gamma(1)*eye(NNroi);
    pbg = [Y1(bg),Y2(bg)];
    Gamma_prior_bg = GaussianDistanceCovariance(pbg,gamma(3),clength(3)) + 0.0001*gamma(3)*eye(NNbg);
end

pobst = [Y1(obst),Y2(obst)];
Gamma_prior_obst = GaussianDistanceCovariance(pobst,gamma(2),clength(2)) + 0.0001*gamma(2)*eye(NNobst);


kaikki = (roi | obst | bg); % in case one wants to leave some pixels out, i.e. ROI, OBST and BG do not cover the whole square 
NN = sum(kaikki); % number of considered pixels (typically N^2, see the previous comment)
ind = find(kaikki);

% place the elements of the built _smaller_ covariance matrices into the
% correct positions in an NNxNN "total covariance matrix"
if roi_bg_correl % ROI and bg have a common prior
    Gamma_Prior_roi_bg = zeros(NN,NN);
    Gamma_Prior_roi_bg(roi(ind) | bg(ind), roi(ind) | bg(ind)) = Gamma_prior_roi_bg;
else % ROI and bg have independent priors
    Gamma_Prior_roi = zeros(NN,NN);
    Gamma_Prior_roi(roi(ind),roi(ind)) = Gamma_prior_roi;
    Gamma_Prior_bg = zeros(NN,NN);
    Gamma_Prior_bg(bg(ind),bg(ind)) = Gamma_prior_bg;
end

Gamma_Prior_obst = zeros(NN,NN);
Gamma_Prior_obst(obst(ind),obst(ind)) = Gamma_prior_obst;

if roi_bg_correl % ROI and bg have a common prior
    Gamma_prior =  Gamma_Prior_roi_bg + Gamma_Prior_obst;
else % ROI and bg have independent priors
    Gamma_prior = Gamma_Prior_roi + Gamma_Prior_nonroi + Gamma_Prior_bg;
end

% Cholesky factor of the prior and its inverse diagonal
Ch = chol(Gamma_prior);

% The following are only needed for D-optimality. Most probaly some of
% these are non-optimal or computed for a second time in this script.
if D_optimality
    dCh = 1./diag(Ch);

    % the inverse covariance for the prior; inv(Gamma_prior) would actually be
    % faster...
    invGamma_prior = Ch\(Ch'\eye(NN)); % the inverse covariance for the prior
    rest = (obst | bg); % pixels that we are _not_ interested in, i.e., do not aim to gather information about 
    ind_rest = find(rest);

    % a matrix needed for evaluating the determinant of the _marginal covariance
    % matrix_ over the ROI
    invGamma_prior_rest = invGamma_prior(ind_rest,ind_rest); 
    Ch_rest = chol(invGamma_prior_rest);    % Could be made faster!!!
    dCh_rest = diag(Ch_rest);
end
    
% visualizing the ROI
figure('position',[0, 350, 450, 450])
pcolor(Y1,Y2,real(ROI));
title('ROI')
axis square, shading flat
drawnow


%% The projection matrix including all angles

if form_xray

    display('Forming the X-ray projection matrix')
    
    % The X-ray projection matrix: All combinations of the lines s and angles
    % phi are initially considered. Subsequently, a row for the final
    % projection matrix, i.e., a row corresponding to a single sensor, is
    % obtained by taking the mean of mm consecutive ows. In kokoX, the rows corresponding
    % to a single projection angle are consecutive. In particular, length(s)
    % must be divisible by mm -- as it is in this script (nn = mm*n).
    kokoX = Xraymat_sum_rows(s, phi, N, mm);
    
    display('Done')
    
    X = kokoX; % pick only the columns that correspond to the considered part of the square
    XT = X'; % many manipulations are faster for columns than
    % indices of detectors that have nonobstructed view of vision
    vis_ind = ( sum(XT(obst,:)) == 0 );  % it is faster to sum values in columns of a sparse matrix in matlab
    save Xray_matriisit X vis_ind


    % only needed for D-optimality
    if D_optimality
        Xrest = X(:,rest); % only the columns corresponding to pixels we are not interested in
        XrestT = Xrest';
        save("Xray_matriisit", "Xrest", "-append") 
    end
    

else

    load Xray_matriisit
    XT = X';
    if D_optimality
        XrestT = Xrest';
    end
end


%% Define the standard deviation of noise

var = sigma^2*ones(n*m,1); % a very simple noise model: the noise variance for all measurements is sigma^2
Var = reshape(var,n,m); 


%% Example draws from the prior

% making four draws from the prior
if draws_from_posterior
    fig2 = figure;
    movegui(fig2,'southeast');
    for j=1:4
        white_noise = randn(NN,1);
        sample = Ch'*white_noise;
        proj = X*sample; 
        Sample = zeros(N^2,1);
        Sample(kaikki) = sample;
        Sample = reshape(Sample,N,N);
        subplot(2,2,j);
        imagesc(Sample)
        % the largest value in the projections corresponding to the randomly
        % drawn image
        title(num2str(max(abs(proj))));
        axis square, colorbar
    end
    movegui(fig2,'southeast');
end

drawnow


%% Iterating over the number of projections

Gamma = Gamma_prior;

if D_optimality
    % some auxiliary matrices and vectors for storing things
    invGamma = invGamma_prior;
    invGamma_rest = invGamma_prior_rest;
    CC = chol(invGamma);
    CC_rest = chol(invGamma_rest);
    dCC = diag(CC);
    dCC_rest = diag(CC_rest);
    doptimi = [];
    dind = [];
    DOptimi = doptimi; 
    DInd = dind; 
end

if A_optimality
    % some auxiliary matrices and vectors for storing things
    aoptimi = [];
    aind = [];
    AOptimi = aoptimi;
    AInd = aind;
end

for k=1:n_dir
    Dopt = [];
    Aopt = [];
    
    if A_optimality
        % Weight accounts for the ROI
        Gamma_w = Gamma*Weight; % A-weighted prior, i.e., posterior from previous step
        apu = diag(Gamma);
        Abaseline = sum(apu(roi)); % A-weighted trace of the prior
    end
       
    str = ['Computing the ' num2str(k) 'th optimality target as a function of the angle'];
    display(str)
    
    % The exhaustive search: loops over projection angles and lateral
    % positions
    for j=1:m
        for jj=1:np
            
            % the rows corresponding to the jth projection angle and jjth
            % position, accounting only for nonobstructed detectors
            tempXT = XT(:,(j-1)*n+jj:j*n-np+jj); % the potentially acceptable rows
            indeksit = vis_ind((j-1)*n+jj:j*n-np+jj); % the indices of nonobstructed rows       
            tempXT = tempXT(:,indeksit);  
            tempX = tempXT';
            
            temp_var = var((j-1)*n+jj:j*n-np+jj);
            Gamma_noise = diag(temp_var(indeksit)); % noise variance for the considered measurements
            
            if D_optimality

                % See the numerical implementation in 
                % "M. Burger, A. Hauptmann, T. Helin, N. Hyvönen, and J.-P. Puska, 
                % Sequentially optimized projections in X-ray imaging, 
                % Inverse Problems, 37, 075006 (2021)".
                tempXrestT = XrestT(:,(j-1)*n+jj:j*n-np+jj); % the collumns corresponding to noninteresting pixels
                tempXrestT = tempXrestT(:,indeksit);
                Apu = CC'\tempXT;
                Apu_rest = CC_rest'\tempXrestT;
                B = Gamma_noise + Apu'*Apu;
                C = chol(B); % a Cholesky factor
                B_rest = Gamma_noise + Apu_rest'*Apu_rest; % the same for the noninteresting pixels
                C_rest = chol(B_rest);
                dC = diag(C); % diagonal of the Cholesky
                dC_rest = diag(C_rest);
        
                % The _to-be-maximized_ target, i.e., the increase in the Shannon information about the ROI 
                % compared to the prior (the last four terms do not depend on the angle). Note that
                % sum(log(dCh)) - sum(log(dCh_rest)) is subtracted (comparison to the prior). 
                % The terms sum(log(dCC_rest)) and sum(log(dCC)) originate from the matrix determinant lemma.
                dopt = -full(sum(log(dC_rest)) - sum(log(dC)) + sum(log(dCC_rest)) - sum(log(dCC)) + sum(log(dCh)) - sum(log(dCh_rest)));  
                Dopt = [Dopt, dopt]; % save the result
            end
            
            if A_optimality
                tempX = tempXT';
                
                % See the numerical implementation in 
                % "M. Burger, A. Hauptmann, T. Helin, N. Hyvönen, and J.-P. Puska, 
                % Sequentially optimized projections in X-ray imaging, 
                % Inverse Problems, 37, 075006 (2021)".
                % matrix that needs to be inverted to get the posterior covariance
                B = Gamma_noise + tempX*Gamma*tempXT;
                C = chol(B);
                Apu = C'\tempX*Gamma_w;
                
                % weighted trace of the posterior
                aopt = Abaseline - sum(sum(Apu.^2));
 
                Aopt = [Aopt, aopt]; % save the result
            end
        end
        
        % monitoring the progress of the exhaustive search
        if mod(j,10)==0
            display([k,n_dir,j,m])
        end
        
    end

    display('Done')
    
    if D_optimality
        [doptimi,dind] = max(Dopt);  % The optimal value and its index
        indj = ceil(dind/np);
        indjj = mod(dind-1,np)+1;
        Dopt_plot = reshape(Dopt,np,m)';
    
        % store all optimality plots in a single figure
        fig = figure(333);
        hold on
        plot(phi,Dopt_plot)
        plot(phi(indj), doptimi,'*')
        str = [num2str(k+1) ' first D-optimality targets as a function of the projection angle. Stars denote the chosen angles.'];
        title(str)
        movegui(fig,'southwest');
    
        % saving and displaying stuff
        DOptimi = [DOptimi doptimi];
        DInd = [DInd dind];
        display([DOptimi; 180/pi*phi(ceil(DInd/n))])
    
        % forming next the new inverse posterior for the optimal angle
        
        tempXT = XT(:,(indj-1)*n+indjj:indj*n-np+indjj); % it is faster to manipulate columns
        indeksit = vis_ind((indj-1)*n+indjj:indj*n-np+indjj);
        tempXT = tempXT(:,indeksit);
        tempX = tempXT';
    
        % the inverse posterior covariance after the measurement
        temp_var = var((indj-1)*n+indjj:indj*n-np+indjj);
        invGamma_noise = diag(1./temp_var(indeksit)); % noise inverse covariance for the optimal angle
        invGamma = invGamma + tempXT*invGamma_noise*tempX;
    
        CC = chol(invGamma);
        dCC = diag(CC);
        CC_rest = chol(invGamma(ind_rest,ind_rest));
        dCC_rest = diag(CC_rest);
    
        % example draws from the 'current posterior'
    
        if draws_from_posterior
            fig2 = figure;
            movegui(fig2,'southeast');
            for j=1:4
                white_noise = randn(NN,1);
                sample = CC\white_noise;
                proj = X*sample; 
                Sample = zeros(N^2,1);
                Sample(kaikki) = sample;
                Sample = reshape(Sample,N,N);
                figure(fig2);
                subplot(2,2,j);
                imagesc(Sample)
                % the largest value in the projections corresponding to the randomly
                % drawn image
                title(num2str(max(abs(proj))));
                axis square, colorbar
            end
        end
    
        if visualize_variances  
            figure('Position',  [940, 550, 450, 450])
            Apu = inv(invGamma);
            apu = diag(Apu);
            kokoVari = zeros(N^2,1);
            kokoVari(kaikki) = apu;
            KokoVari = reshape(kokoVari,N,N);
            pcolor(Y1,Y2,sqrt(KokoVari))
            hold on
            
            % plotting the parallel beams
            t = linspace(-1,1,2);
            ss1 = s((indjj-1)*mm+1);
            x1 = ss1*cos(phi(indj)) - t*sin(phi(indj));
            y1 = -(ss1*sin(phi(indj)) + t*cos(phi(indj)));
            plot(x1,y1, 'k', 'Linewidth', 3)
        
            ss2 = s((indjj-1+nr)*mm);
            x2 = ss2*cos(phi(indj)) - t*sin(phi(indj));
            y2 = -(ss2*sin(phi(indj)) + t*cos(phi(indj)));
            plot(x2,y2, 'k', 'Linewidth', 3)
            
            axis square, colorbar, shading flat, caxis([0,gamma(1)])
            title(['STD after ' num2str(k) 'th memeasurement (D)'])
        end
    end
    
    if A_optimality
    
        [aoptimi,aind] = min(Aopt);  % The optimal value and its index
        indj = ceil(aind/np);
        indjj = mod(aind-1,np)+1;
        Aopt_plot = reshape(Aopt,np,m)';
    
    
        % store all optimality plots in a single figure
        fig = figure(444);
        hold on
        plot(phi,Aopt_plot)
        plot(phi(indj), aoptimi,'*')
        str = [num2str(k+1) ' first A-optimality targets as a function of the projection angle. Stars denote the chosen angles.'];
        title(str)
        movegui(fig,'southwest');
    
        % saving and displaying stuff
        AOptimi = [AOptimi aoptimi];
        AInd = [AInd aind];
        display([AOptimi; 180/pi*phi(ceil(AInd/n))])
    
        % forming the new inverse posterior for the optimal angle    
        
        tempXT = XT(:,(indj-1)*n+indjj:indj*n-np+indjj); % it is faster to manipulate columns
        indeksit = vis_ind((indj-1)*n+indjj:indj*n-np+indjj);
        tempXT = tempXT(:,indeksit);
        tempX = tempXT';

        temp_var = var((indj-1)*n+indjj:indj*n-np+indjj);
        Gamma_noise = diag(temp_var(indeksit)); % noise inverse covariance for the optimal angle
        invB = inv(Gamma_noise + tempX*Gamma*tempXT);
        Gamma = Gamma-(Gamma*tempXT)*invB*(tempX*Gamma);

        % example draws from the 'current posterior'

        if draws_from_posterior
            CCC = chol(Gamma);
            fig2 = figure;
            movegui(fig2,'southeast');
    
            for j=1:4
                white_noise = randn(NN,1);
                sample = CCC'*white_noise;
                proj = X*sample; 
                Sample = zeros(N^2,1);
                Sample(kaikki) = sample;
                Sample = reshape(Sample,N,N);
                figure(fig2)
                subplot(2,2,j);
                imagesc(Sample)
                % the largest value in the projections corresponding to the randomly
                % drawn image
                title(num2str(max(abs(proj))));
                axis square, colorbar
            end
            movegui(fig2,'southeast');
        end
    
        if visualize_variances
            figure('Position',  [940, 550, 450, 450])    
            apu = diag(Gamma);
            kokoVari = zeros(N^2,1);
            kokoVari(kaikki) = apu;
            KokoVari = reshape(kokoVari,N,N);
            pcolor(Y1,Y2,sqrt(KokoVari))
            hold on
            
            % the parallel beams
            t = linspace(-1,1,2);
            ss1 = s((indjj-1)*mm+1);
            x1 = ss1*cos(phi(indj)) - t*sin(phi(indj));
            y1 = -(ss1*sin(phi(indj)) + t*cos(phi(indj)));
            plot(x1,y1, 'k', 'Linewidth', 3)
        
            ss2 = s((indjj-1+nr)*mm);
            x2 = ss2*cos(phi(indj)) - t*sin(phi(indj));
            y2 = -(ss2*sin(phi(indj)) + t*cos(phi(indj)));
            plot(x2,y2, 'k', 'Linewidth', 3)
            
            axis square, colorbar, shading flat, caxis([0,gamma(1)])
            title(['STD after the ' num2str(k) 'th measurement (A)'])
        end
    
    end
    
    drawnow
    
    if pauseko
        display('Hit a key to continue')
        pause
    end
    
end

if D_optimality
    save doptimal_b DInd pr n m n_dir sigma roi_bg_correl gamma clength expt ROI OBST BG
end

if A_optimality
    save aoptimal_b AInd pr n m n_dir sigma roi_bg_correl gamma clength expt ROI OBST BG
end

%% Plotting optimal angles in human-readable units (in degrees between 0 and 180)
% 
% if D_optimality
%     Phi_Dopt = 180/pi*phi(ceil(DInd/np));
%     display([Phi_Dopt; 1:n_dir])
% 
%     figure, plot(1:n_dir,Phi_Dopt,'*')
%     ylim([0,180])
%     title('The D-optimal projection angles')
%     xlabel('Iteration number')
%     ylabel('Optimal angle in degrees')
% end
% 
% if A_optimality
%     Phi_Aopt = 180/pi*phi(ceil(AInd/np));
%     display([Phi_Aopt; 1:n_dir])
%     figure, plot(1:n_dir,Phi_Aopt,'*')
%     ylim([0,180])
%     title('The A-optimal projection angles')
%     xlabel('Iteration number')
%     ylabel('Optimal angle in degrees')
% end
