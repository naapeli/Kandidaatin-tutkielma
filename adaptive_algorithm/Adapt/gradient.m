function [Reunat, gradientti] = gradient(Kuva)

% Kuva -- square image
%
% Reunat = absolute value of the gradient as an image of the same size

% central difference approximation for the derivative 
N = size(Kuva,1);
diagonals = 0.5*[-ones(N,1) ones(N,1)];
diag_places = [-1,1];
D = spdiags(diagonals, diag_places, N, N);
% D(1,1) = -0.5; D(end,end) = 0.5; % assuming homogeneous Neumann condition

% the first and last line corrected as forward and backward differences
D(1,1) = -1; D(1,2) = 1;
D(end,end-1) = -1; D(end,end) = 1;

% proper scaling(?)
h = 1/(N+1);
D = D/h;

% the two dimensional derivatives as kronecker products
I = speye(N);
DD1 = kron(I,D);
DD2 = kron(D,I);

% the two derivatives as a taller matrix
DD = [DD1; DD2];

% computing the absolute value of the gradient
gradientti = DD*Kuva(:); % the derivatives as a single vector
reunat2 = reshape(gradientti,N^2,2); % the two columns correspond to the two directions
Reunat = sqrt(sum(reunat2.^2,2)); % length of the gradien at each pixel
Reunat = reshape(Reunat,N,N); % back to the original size

end

