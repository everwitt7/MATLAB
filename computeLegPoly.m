function [ z ] = computeLegPoly( x, Q )
%COMPUTELEGPOLY Return the Qth order Legendre polynomial of x
%   Inputs:
%       x: vector (or scalar) of reals in [-1, 1]
%       Q: order of the Legendre polynomial to compute
%   Output:
%       z: matrix where each column is the Legendre polynomials of order 0 
%          to Q, evaluated at the corresponding x value in the input

    z = zeros(length(x), Q+1);
    % need an extra space for Legendre polynomial of 0, and set the column
    % equal to 1
    z(:,1) = 1;
    
    % need to iterate from 1 to length(x) for rows
    % need to iterate from 0 to Q+1 for columns
    % each x will be different, so compute each legendre sequentially
    for row = 1:length(x)
        % computing the Legendre polynomial of 1 and then iterativley
        % calculating the values from 3 to Q+1
        z(row, 2) = x(row);
        for col = 3:Q+1
            pp = ( ( 2 * col - 1 ) / col ) * x(row) * z(row, col - 1);
            p = ( ( col - 1 ) / col ) * z(row, col - 2);
            z(row, col) = pp - p;
        end
    end
end