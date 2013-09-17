% =======================================================================
% 
% Particle Bayesian Optimisation for Parameter Inference
% Hull-White Stochastic Volaility model
%
% Reproduces example in the poster presentation for
% Dahlin, J. and Lindsten, F. Bayesian Optimisation for Parameter Inference
%
% presented at ERNSI Wotrkshop, Nancy, France, September, 2013.
%
% Copyright (c) 2013 Johan Dahlin [ johan.dahlin (at) liu.se ]
% Date: 2013 - 09 -17
%
% Description: Main function, just run and it will produce a plot similiar
%              to the poster example.
%
% =======================================================================

clear all; close all;
rng(128,'twister');

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

par.Npart       = 500;               % Number of particles
par.xo          = 0;                 % Initial state
par.Po          = 0.0001;            % Initial covariance (used in Kalman)
par.nIter       = 150;               % Number of iterations
par.nInitial    = 10;                % Number of initial samples (random)
par.philimit    = 1;                 % Maximum abs value of phi
par.sigmalimit  = 2;                 % Maximum value of sigma (min is 0)
par.epsilon     = 0.01;              % Exploration vs. Exploitation coef.
                                     % sensitive but 0.01 is a standard
                                     % value according to Lizotte (2008)

% Specify functions
meanfunc        = @meanConst;               % Constant mean function
covfunc         = {@covMaterniso, 3};       % Mantern covariance function
likfunc         = @likGauss;                % Gaussian likelihood

% Specify priors (overwritten by Emperical Bayes function later)
hyp.mean        = 1;
hyp.cov         = log([7; 5]);
hyp.lik         = log(0.1);                 % Standard deviation of noise


% =======================================================================
% % Generate data without input
% =======================================================================

data            = datagen(sys,zeros(sys.T,1));


% -------------------------------------------------------------------
% Generate a number of inital parameter sets
% -------------------------------------------------------------------    

th       = sys;
ths(1:par.nInitial,:) = [ par.philimit * 2 *( rand(par.nInitial,1) - 0.5)...
                          par.sigmalimit * rand(par.nInitial,1) ];

% -------------------------------------------------------------------
% Generate GP grid
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

% =======================================================================
% Main loop
% =======================================================================
                     
for kk = 1:par.nIter
    
    % -------------------------------------------------------------------
    % Sample the likelihood 
    % -------------------------------------------------------------------   
    
    th.phi      = ths(kk,1);
    th.sigmav   = ths(kk,2);
    ll(kk)      = pf(data,sys,th,par);

    % -------------------------------------------------------------------
    % Fit the GP
    % -------------------------------------------------------------------
    
    if (kk >= par.nInitial) 
        % Estimate the hyperparameters
        hyp2 = minimize(hyp, @gp, -50, @infExact, meanfunc, covfunc, likfunc, ths, ll');

        % Fit the GP to the data on the grid
        [m,s2] = gp(hyp2, @infExact, meanfunc, covfunc, likfunc, ths, ll', t);
    end

    % -------------------------------------------------------------------
    % Aquire new parameters
    % -------------------------------------------------------------------

    if (kk >= par.nInitial) 
        % Find x+
        [llmax tmp] = max(m);
        thmax       = t(tmp,:);

        % Calculate auxillary quantity
        s   = sqrt( s2(:,end) );
        tmp = m - llmax - par.epsilon;
        PZ  = normcdf( ( tmp ./ s ) .* ( s>0 ) );
        pZ  = normpdf( ( tmp ./ s ) .* ( s>0 ) );
        EI  = tmp.*PZ + s.*pZ;

        % Find the maximising argmument of PI
        [EImax Zmax] = max(EI); 
        ths(kk+1,:)  = t(Zmax,:);
        
        % Check if changes
        %if ( ths(kk+1,:) == ths(kk,:) )
        %    kk = par.nIter;
        %    break;
        %end
    end

    % -------------------------------------------------------------------
    % Print progress
    % -------------------------------------------------------------------
    
    if (rem(kk,10)==0)
        disp(['=============== iteration ' num2str(kk) ' complete. ===============']); 
    end
end

% =======================================================================
% Plot the results
% =======================================================================

ms = reshape(m, length(t1), length(t1));

figure(1);
clf;
subplot(111); surface(t1,t2,ms,'EdgeColor','none');
              hold on; plot(ths(:,1),ths(:,2),'k.');
              xlabel('theta_1'); ylabel('theta_2');

                           
[tmp,thmax1] = max(ms);
[tmp,thmax2] = max(tmp);
thOpt = [t1(thmax2) t2(thmax1(thmax2))];

disp(['=============== algoritm complete. ===============']); 
disp(['estimated parameters. phi: ' num2str(thOpt(1)) ' and sigma: ' num2str(thOpt(2)) ]);
disp(['true parameters.      phi: ' num2str(sys.phi) ' and sigma: ' num2str(sys.sigmav) ]);


%save('svmodel.mat','t1','t2','m','s2','t','ms','ths','thOpt','-v6')


% =======================================================================
% End of file
% =======================================================================