% =======================================================================
% 
% Particle Bayesian Optimisation for Parameter Inference
% Hull-White Stochastic Volaility model
%
% Reproduces example in 
% J. Dahlin and F. Lindsten, 
% Particle filter-based Gaussian Process Optimisation for Parameter Inference. 
% Proceedings of the 18th World Congress of the International Federation of 
% Automatic Control (IFAC), Cape Town, South Africa, August 2014. 
% (submitted, pending review) 
%
% Copyright (c) 2013 Johan Dahlin [ johan.dahlin (at) liu.se ]
% Date: 2013 - 11 -29
%
% Description: Main function, just run and it will produce a plot similiar
%              to the example in the paper.
%
% =======================================================================

clear all; close all;

% =======================================================================
% System model and parameters
% x(t+1) = sys.f + sys.fu + sys.fn * v(t),  v(t) ~ N( 0, 1 )
% y(t)   = sys.g + sys.gu + sys.gn * e(t),  e(t) ~ N( 0, 1 )
% =======================================================================

sys.f           = @(sys,x,t) sys.phi.*x;
sys.fu          = @(sys,u,t) 0;
sys.fn          = @(sys,x,t) sys.sigmav;
sys.g           = @(sys,x,t) 0;
sys.gu          = @(sys,u,t) 0;
sys.gn          = @(sys,x,t) sys.beta .* exp(x./2);

% System parameters
sys.sigmav      = 0.16; 
sys.phi         = 0.98;  
sys.beta        = 0.65;
sys.T           = 500;      
sys.xo          = 0;


% =======================================================================
% Algorithm parameters
% =======================================================================

par.Npart       = 2000;              % Number of particles
par.xo          = 0;                 % Initial state
par.Po          = 0.0001;            % Initial covariance (used in Kalman)
par.nInitial    = 20;                % Initial number of random samples
par.nIter       = 100;               % Number of iterations
par.philimit    = 0.999;             % Maximum abs value of phi
par.sigmalimit  = 2;                 % Maximum value of sigma (min is 0)
par.epsilon     = 0.01;              % Exploration vs. Exploitation coef.
                                     % sensitive but 0.01 is a standard
                                     % value according to Lizotte (2008)
opts.maxevals   = 500;
opts.maxits     = 500;
opts.showits    = 0;
bounds          = [-par.philimit, par.philimit; 0 par.sigmalimit];

% Specify functions
meanfunc        = {@meanSum, {@meanLinear, @meanConst}};
covfunc         = {@covMaterniso, 3};
likfunc         = @likGauss;

% Specify priors (overwritten by Emperical Bayes function later)
hyp0.mean       = [-500; 0; 2];
hyp0.cov        = [3 8];
hyp0.lik        = -1.5;

% Generate data
data            = datagen(sys,zeros(sys.T,1));

% =======================================================================
% Main loop
% =======================================================================

th       = sys;

for kk = 1:par.nInitial
    % -------------------------------------------------------------------
    % Generate a number of inital parameter sets
    % -------------------------------------------------------------------    

    ths(kk,:)   = [ par.philimit * 2 * (rand - 0.5) par.sigmalimit * rand];
    th.phi      = ths(kk,1);
    th.sigmav   = ths(kk,2);
    ll(kk)      = pf(data,sys,th,par);
end

for kk = par.nInitial:par.nIter
    % -------------------------------------------------------------------
    % Sample the likelihood 
    % ----------------------------------------------------------------
    
    th.phi      = ths(kk,1);
    th.sigmav   = ths(kk,2);
    ll(kk)      = pf(data,sys,th,par);

    % -------------------------------------------------------------------
    % Fit the GP hyper parameters
    % -------------------------------------------------------------------
    
    hyp = minimize(hyp0, @gp, -100, @infExact, meanfunc, covfunc, likfunc, ths, ll');

    % -------------------------------------------------------------------
    % Find the value of \mu_{\max}
    % ------------------------------------------------------------------- 
    
    % Fit the GP to the data and extract the new mean function
    [m,s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, ths, ll', ths);
    
    % Compute the maximising element in ths of mu
    [~,nmax] = max(m);
    
    % Extract the maximum observed ll-estimate and its parameters
    llmax(kk)   = m(nmax); % ll(nmax);
    thmax(kk,:) = ths(nmax,:);
    
    % -------------------------------------------------------------------
    % Aquire new parameters using Direct optimisation of EI
    % -------------------------------------------------------------------
    
    criteria = @EI; Problem.f = criteria;
    [minval, newX, hist] = Direct(Problem, bounds, opts, ths, ll , hyp,...
                                  meanfunc, covfunc, likfunc, llmax(kk),...
                                  par.epsilon);
    % Save the EI value
    eis(kk) = minval;
                              
    % Determine the next point in which to sample the log-likelihood
    ths(kk+1,:) = newX' + 0.02 * ( rand(1,2) - 0.5 );

    % -------------------------------------------------------------------
    % Print progress
    % -------------------------------------------------------------------
    
    if (rem(kk,10)==0)
        disp(['=============== iteration ' num2str(kk) ' complete. ===============']); 
    end
end

% Run DIRECT for thetahat
opts.maxevals = 1000; opts.maxits = 1000; opts.showits = 0;
criteria = @evalMu; Problem.f = criteria;
[minval, thhat, hist] = Direct(Problem, bounds, opts, ths(1:kk,:),...
                               ll, hyp, meanfunc, covfunc, likfunc);

                          
% =======================================================================
% Plot the results
% =======================================================================

% -------------------------------------------------------------------
% Generate grid for plotting
% -------------------------------------------------------------------                 

t1    = linspace(-1,1,301); 
t2    = linspace(0.01,2,301);
kk    = 1;

for ii = t1
    for jj = t2
        t(kk,:) = [ii jj];
        kk = kk + 1;
    end
end

[m,s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, ths(1:par.nIter,:), ll', t);
ms = reshape(m, length(t1), length(t2));

figure(1);
    clf;
    surface(t1,t2,ms,'EdgeColor','none');
    hold on; 
        plot(ths(:,1),ths(:,2),'k.');
        xlabel('theta_1'); ylabel('theta_2');
    hold off;

clc;
disp(['=============== algoritm complete. ===============']); 
disp(['estimated parameters. phi: ' num2str(thhat(1)) ' and sigma: ' num2str(thhat(2)) ]);
disp(['true parameters.      phi: ' num2str(sys.phi) ' and sigma: ' num2str(sys.sigmav) ]);

% =======================================================================
% End of file
% =======================================================================   
