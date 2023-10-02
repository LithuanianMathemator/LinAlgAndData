function result = svdclass(A, T)
    % A: n x 784 matrix containing n digits
    % T: 784 x k x 10 matrix containing first k singular vectors for each
    % of the training sets

    % output: n x 1 vector containing labels (0-9) for digits in A
    
    A = double(A);
    distances = zeros(size(A,1),1);
    result = zeros(size(A,1),1);
    for i=1:size(A,1)
        distances(i) = norm( A(i,:)' - T(:,:,1)*(T(:,:,1)'*A(i,:)') );
        result(i) = 0;
        for k=2:10
            dist = norm( A(i,:)' - T(:,:,k)*(T(:,:,k)'*A(i,:)') );
            if (dist < distances(i))
                distances(i) = dist;
                result(i) = k-1;
            end
        end
    end
end
