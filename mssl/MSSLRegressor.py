#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:54:33 2018

@author: goncalves1
"""

		if strcmp(prob_type,'regression'),
            
            switch( opts_ao.alg_w ),
                
                case 'closed-form',
                    L = kron( eye(dimension), Omega );
                    Xls = x+(1 * P * L * P');
                    Wvec = Xls\y;
                
                case 'fminunc',
                    Wvec = fminunc( @(w)cost_function_squared_loss(w, x, y, Omega, P), Wvec, opt );
                
                case 'fista', 
                    
                    [Wvec,~] = acc_proximal_gradient( Wvec, x, y, Omega, P, opts_ao.gamma, 10 );

                case 'lbfgs',
                    opts.x0 = Wvec;
                    [Wvec, ~, info] = lbfgsb(@(w)cost_function_squared_loss(w, x, y, Omega, P), ...
                                          -inf(ntasks*dimension,1), inf(ntasks*dimension,1), opts );
                
                otherwise
                    error('Optimization algorithm for W-step is not valid!');
                    
                
            end
            
