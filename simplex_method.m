function [] = simplex_method ( c, A, b )

    % c -> the objective vector, column vector, (nx1)
    % A -> the constraint matrix (mxn)
    % b -> RHS of the constraints, column vector (mx1)
    
    % x -> vector of the decision variables, column vector (nx1)
    % s -> vector of slack variables, column vector (mx1)
    % z -> trying to maximize this value, z = transpose(C)*x, (1x1)
    
    disp("Input: c");
    disp(c);
    disp("Input: A");
    disp(A);
    disp("Input: b");
    disp(b);
    
    % # of rows in A will be the # of slack variables
    m = size(A,1);
    
    % # of rows in c will be the # of decision variables
    n = size(c,1);
    
    
    % 1. Initialization:
    
    % define c_tilda to be the concatenation of c and 0, the 0 represents
    % the coefficients in front of the slack variables in the objective
    % function, (n+mx1)
    zero_cat = zeros(m,1);
    c_tilda = vertcat(c, zero_cat);
    
    % define A_tilda to be the concatenation of A and I (mxm Identity)
    % to represent the slack variables added to each constrait inequality
    % to make it a constraint equality (mxm+n)
    identity_cat = eye(m);
    A_tilda = horzcat(A, identity_cat);
    
    % now we have c_tilda and A_tilda in the augmented canonical standard
    % form and can initialize the basis to be the slack variables, so given
    % n decision variables and m slack variables, the basis will be from
    % n+1 to n+m
    
    
    % 2. Define B and c_sub(B)
    
    % B = A_tilda(:,basis) -> B represents all of the rows in the column of 
    % the basis (initially is the identity matrix), basis is n+1:n+m
    B = A_tilda(:,n+1:n+m); 
    
    % c_sub(B) = c_tilda(basis) -> represents the basis (initially all
    % zeros)
    c_subB = c_tilda(n+1:n+m);
    
    % just to keep track of which variables are in the basis currently so
    % that swapping is easier (just should be indexes of decision and slack
    % variables in each vector)
    not_basis = zeros(n,1);
    basis = zeros(m,1);

    for i = 1:n
        not_basis(i) = i;
    end
    
    for i = 1:m
        basis(i) = n+i;
    end
    
    % 3. Initialize c_ideal, A_ideal, and b_ideal, and find Enterting Basic
    % Variable, E
    continuous_loop = true;
    while continuous_loop
        disp("ITERATIONS!");

        % c_ideal = transpose(c_sub(B))*inverse(B)*A_tilda-transpose(c_tilda)
        c_ideal = transpose(c_subB) * B^(-1) * A_tilda - transpose(c_tilda);

        % if min(c_ideal) >= 0, then terminate the program and report the
        % optimal solution and optimal value... x_tilda = inverse(B)b_ideal
        % otherwise make the entering basic variable, E, min(c_ideal)
        % the E_index refers to the COLUMN of A_tilda
        [E_value, E_index] = min(c_ideal);

        if E_value >= 0
            disp("optimal");
            break;
        end

        % A_ideal = inverse(B)*A_tilda
        % b_ideal = inverse(B)*b
        A_ideal = B^(-1) * A_tilda;
        b_ideal = B^(-1) * b;


        % 4. Define the Leaving Nonbasic Variable, L, if bounded

        % if max(A_ideal(:,E_index)) > 0, then we continue, otherwise if it is 
        % <= to zero the LP is ubounded, so terminate and report no solution
        % otherwise L = min(b_ideal/A_ideal(:,E_index)), and A_ideal(:,E_index)
        % values are all greater than 0, L_index refers to the ROW of A_tilda
        current_column = b_ideal ./ A_ideal(:,E_index);
        positive_indexes = current_column > 0;
        
        if sum(positive_indexes) <= 0
            disp("unbounded");
            break;
        end
        
        [~, L_index] = min(current_column(positive_indexes));
        
        % updating the basis/not_basis indexes
        temp_index = not_basis(E_index);
        not_basis(E_index) = basis(L_index);
        basis(L_index) = temp_index;

        
        % 5. Swap E and L, then update B and c_sub(B), then update ideal values 
        % and check min(c_ideal) and repeat until all values > 0
        
        % need to keep track of the basis because A_ideal columns update
        for i = 1:m
            B(:,i) = A_ideal(:,basis(i));
            c_subB(i) = c_tilda(basis(i));
        end        
        
    end
    
    % concatenate the basis and non_basis (in order according to indexes
    % and then calculate the final value using c_tilda and x_tilda)
    optimal_solution = B^(-1) * b_ideal;
    x_tilda = vertcat(basis, not_basis);
    
    optimal_zeros = zeros(n,1);
    optimal_solution = vertcat(optimal_solution,optimal_zeros);
    
    [~, x_tilda_order] = sort(x_tilda);
    new_optimal = optimal_solution(x_tilda_order,:);
    
    optimal_value = transpose(c_tilda) * new_optimal;
    
    disp("Optimal Solution - With Slack Variables");
    disp(new_optimal);
    
    disp("Optimal Value");
    disp(optimal_value);
    
end
