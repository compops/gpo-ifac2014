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
% Description: Calculates the expected improvement for the optimisation
%
% =======================================================================

function ei = EI(xx,ths,ll,hyp2,meanfunc,covfunc,likfunc,llmax,epsilon)

% Fit the GP to the data and extract the new position
[m,s2] = gp(hyp2, @infExact, meanfunc, covfunc, likfunc, ths, ll', xx');

% Calculate auxillary quantites
s   = sqrt(s2(:,end));
yres  = m - llmax - epsilon;
ynorm = (yres/s) * ( s>0 );

% Compute the EI and negate it
ei  = yres * normcdf(ynorm) + s * normpdf(ynorm);
ei  = max([ei 0]);
ei  = -ei;

% =======================================================================
% End of EI
% =======================================================================
