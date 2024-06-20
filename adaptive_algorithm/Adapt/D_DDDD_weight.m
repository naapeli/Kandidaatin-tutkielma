function [L,DD] = D_DDDD_weight(N,weight)

% N -- size of the square images
% gamma -- weight of the 'penalty term', i.e., 1/gamma multiplies the covariance matrix
% weight -- the 'conductivity' in the inverse covariance (on a 2D grid)

% L -- discretization of the differential operator
%
%      -\nabla \cdot (weight \dot \nabla (\cdot))
% 
% with homogeneous Dirichlet condition. NB: L is positive definite (and
% invertible) as such.

% The x and y derivatives with Dirichlet conditions end up being (N+1)N x
% N^2 matrices. Thus the weights need to be suitably stretched.
weight1 = ([weight; zeros(1,N)] + [zeros(1,N); weight])/2; 
weight2 = ([weight zeros(N,1)] + [zeros(N,1) weight])/2;

% The actual weights as sparse diagonal matrices
Weight1 = spdiags(sqrt(weight1(:)),0,N*(N+1),N*(N+1));
Weight2 = spdiags(sqrt(weight2(:)),0,N*(N+1),N*(N+1));

% one dimensional first derivative with homogeneous Dirichlet conditions;
% (N+1) x N matrix. 
diagonals = [-ones(N,1) ones(N,1)];
diag_places = [-1,0];
D = spdiags(diagonals, diag_places, N+1, N); 
h = 1/(N+1);
D = D/h;

% The two dimensional derivatives as Kronecker products
I = speye(N);
DD1 = kron(I,D);
DD2 = kron(D,I);

% The two dimensional derivatives as a single matrix, with the appropriate
% weights corresponding to the "conductivity".
DD = [Weight1*DD1; Weight2*DD2];

% The discretized differential operator as a sparse matrix could be formed
% as follows
L = DD'*DD;

% NB:
%
% - \int v (\nabla \dot (weight \nabla u) dx = 
% \int (\sqrt(weight) \nabla v) \dot (\sqrt(weight) \nabla u) dx
% \approx \int (DD v)' * (DD u) dx = \int v' * (DD'*DD)*u dx
% 
% if v has homogeneous Dirichlet data.

end

