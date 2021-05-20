function a3_00000000
% Function for CISC271, Winter 2021, Assignment #3

    % Read file and remove first column
    filename = 'wine.csv';
    data = csvread(filename, 0, 1)';
    
    % Set yvec to first column and remove column from data
    yvec = data(:, 1);
    data(:, 1)=[];
    
    % Initialize score matrices
    dbscores = [];
    dbindexes = [];
    
    % Find DB score for each 2 column submatrix of the data
    for i=1 : size(data, 2)
        for j=i+1 : size(data, 2)
            Xmat = data(:, [i j]);
            dbscores = [dbscores dbindex(Xmat, yvec)];
            dbindexes = [dbindexes; [i j]];
        end
    end

    % Find lowest score and variable indexes
    lowest_dbi = find(dbscores == min(dbscores));
    c_idxs = dbindexes(lowest_dbi, :);
    
    % Display lowest score and variable indexes
    disp('Data Columns DB index: ')
    disp(dbscores(lowest_dbi))
    disp('Variables: ')
    disp(c_idxs)
    
    % Plot the clusters
    avec = data(:, c_idxs(1));
    bvec = data(:, c_idxs(2));
    f1 = figure;
    gscatter(avec, bvec, yvec)
    title('Clustering of Best Pair of Values')
    xlabel('Ethanol')
    ylabel('Flavanoids')
    
    
    
    % Calculate zero mean matrix
    zero_mean = data - ones(length(data), 1) * mean(data, 1);


    % Calculate SVD of the data
    [U, S, V] = svd(zero_mean, 0);

    % Find the first 2 right singular vectors and find the DBI
    Zscores = zero_mean * V(:, [1 2]);
    disp('Raw PCA Score: ')
    disp(dbindex(Zscores, yvec))
    db_reduced = [dbindex(Zscores, yvec)];
    
    % Plot the clusters
    f2 = figure;
    gscatter(Zscores(:, 1), Zscores(:, 2), yvec)
    title('Clustering after Reducing Dimensionality')
    xlabel('S1')
    ylabel('S2')
    
    % Standardize the data
    data = zscore(data);
    
    % Calculate zero mean matrix
    zero_mean = data - ones(length(data), 1) * mean(data, 1);

    % Calculate SVD of the data
    [U, S, V] = svd(zero_mean, 0);

    % Find the first 2 right singular vectors and find the DBI
    Zscores = zero_mean * V(:, [1 2]);
    
    disp('Standardized PCA Score: ')
    disp(dbindex(Zscores, yvec))
    db_standardized = [dbindex(Zscores, yvec)];
    
    % Plot the clusters
    f3 = figure;
    gscatter(Zscores(:, 1), Zscores(:, 2), yvec)
    title('Clustering after Standardizing the Data')
    xlabel('S1')
    ylabel('S2')

end
function score = dbindex(Xmat, lvec)
% SCORE=DBINDEX(XMAT,LVEC) computes the Davies-Bouldin index
% for a design matrix XMAT by using the values in LVEC as labels.
% The calculation implements a formula in their journal article.
%
% INPUTS:
%        XMAT  - MxN design matrix, each row is an observation and
%                each column is a variable
%        LVEC  - Mx1 label vector, each entry is an observation label
% OUTPUT:
%        SCORE - non-negative scalar, smaller is "better" separation

    % Anonymous function for Euclidean norm of observations
    rownorm = @(xmat) sqrt(sum(xmat.^2, 2));

    % Problem: unique labels and how many there are
    kset = unique(lvec);
    k = length(kset);

    % Loop over all indexes and accumulate the DB score of each cluster
    % gi is the cluster centroid
    % mi is the mean distance from the centroid
    % Di contains the distance ratios between IX and each other cluster
    D = [];
    for ix = 1:k
        Xi = Xmat(lvec==kset(ix), :);
        gi = mean(Xi);
        mi = mean(rownorm(Xi - gi));
        Di = [];
        for jx = 1:k
            if jx~=ix
                Xj = Xmat(lvec==kset(jx), :);
                gj = mean(Xj);
                mj = mean(rownorm(Xj - gj));
                Di(end+1) = (mi + mj)/norm(gi - gj);
            end
        end
        D(end+1) = max(Di);
    end

    % DB score is the mean of the scores of the clusters
    score = mean(D);
end
