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
% Description: Evaluates the posterior mean function for the optimisation
%
% =======================================================================

function [m] = evalMu(xx,ths,ll,hyp2,meanfunc,covfunc,likfunc)
    [m,s2] = gp(hyp2, @infExact, meanfunc, covfunc, likfunc, ths, ll', xx');
    m = -m;
    sciL = m - 1.96 * sqrt(s2);
    sciH = m + 1.96 * sqrt(s2);
end

% =======================================================================
% End of evalMu
% =======================================================================
