function final_prox = prox_ALsP(c, lambda, base)
% compute the proximal operator of c

p = length(c);      % p is the length of beta
betaws = rand(size(c));       % initial beta via warm start
lenmin = 1e-30;      % step length minimum
lenmax = 1e5;        % step length maximum
len = 0.0001;        % starting steo length


gtol = 1e-10;        % tolerance of gradient is used as the termination criteria
ftol = 1e-9;        % tolearance of function value is used as the termination criteria



B = [eye(p), -eye(p);
    -eye(p), eye(p)];   % the auxilary matrix as in paper


d = [-c;
    c];           % the auxilary vector as in paper


% a>=b := 1 or 0
z = [betaws.*(betaws >= 0);
    (-betaws).*(-betaws >= 0)]; % z = [u; v], u = beta+, v = -beta+

rho = 0.5;           % backtracking line search parameter

fit.break = 0;       % initial output struct


% define f(x) and gradient of f(x)
v = @(z) z(1 : p) + z(p+1 : p+p); % v = alpha * (z1+z(p+1), ... , zp+z(2p))
penalty = @(v)log( sum( base.^v ) ) / log(base); % logab formula
f = @(x) 0.5*x'*B*x + d'*x + lambda * penalty( v(x) ); % objective function as in paper
gf = @(x) B*x + d + lambda  * [ base.^v(x) ; base.^v(x) ] / sum( base.^v(x) );

% % define f(x) and gradient of f(x)
% v = @(z) alpha * (z(1:p) + z(p+1: 2*p)); % v = alpha * (z1+z(p+1), ... , zp+z(2p))
% f = @(x) 0.5*x'*B*x + d'*x + lambda * log( sum( exp( v(x) ) ) ); % objective function as in paper
% gf = @(x) B*x + d + lambda * alpha * [exp( v(x) ); exp( v(x))] / sum( exp( v(x) ) );


fit.funcValue = f(z);   % function value at start point z
gfz = gf(z);            % gradient at start point z


for iter = 1 : 1000     % max iterations
    for q = 0 : 100     % max backtrackin steps 
        znew = (z - rho^q * len * gfz) .* ((z - rho^q * len *gfz) >= 0); % gradient descent
        if f(znew) < f(z)
            fit.funcValue = cat(2, fit.funcValue, f(znew));
            break;
        end  % ensure the new function value is decreased (Armijo condition)
        % else descrease the stepsize p, p^2, ...
    end
    
    if sum(znew < 0) == 0 && abs(f(znew) - f(z)) < ftol
        break; 
     end   % function value almost does not change

    
    deltaz = znew - z;  % descent direction
    gf_znew = gf(znew); % gradient of new point
    
    % avoid the step len too small or large
    len = median( [lenmin, lenmax,  deltaz' * deltaz / ( (gf_znew - gfz)' * deltaz)]); % update the length
    
    
    if sum(znew < 0) == 0 ...     % can be deleted
            && norm(deltaz) < gtol % z almost dose not change
        fit.break = 1;
        break;
    end
    
    z = znew;       % update z
    gfz = gf_znew;  % update the gradient of z        
end

fit.z = z;  % z value
fit.beta = z(1 : p) - z(p+1 : p+p); % final estimate for beta
fit.nonzero = length( find( fit.beta ~= 0)); % number of nonzero element
fit.iter = iter; % number of iteration

final_prox =fit.beta;
end

