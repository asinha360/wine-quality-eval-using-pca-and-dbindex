function a3_20292366
% Function for CISC271, Winter 2022, Assignment #3

    % Imports wine.csv as data using csvread whilst ignoring the first
    % column
    data = csvread('wine.csv', 0, 1);
    
    % Extracts the data matrix as Xmat, the Y label vector as yvec from
    % data and then transposes them for easier analysis and
    % interpretation. Xmat here contains all the data points of the 3
    % cultivars of wine and yvec contains there respective cultivar labels
    % which aids in clustering them on a graph
    Xmat = data(2:end, :);
    yvec = data(1, :);
    Xmat = Xmat';
    yvec = yvec';

    % This is part (a) of the problem where we iterate through the Xmat,
    % which contains all the raw cultivar statistics, finding the DB index 
    % of all possible pairs of columns, and then finding the pair with the 
    % lowest possible DB index
    
    % min_score is the largest value possible as to keep the algorithm
    % bug-free as all possible datapoints are going to be compared against
    % it, while the best pair vector stores the best possible pair ie. the
    % pair with the lowest DB index. This allows for reduction to the most
    % important dimensions during further analysis, as it ensures that our
    % clustering results will be meaningful
    
    min_score = inf;
    best_pair = [];
    for i = 1:12
        for j = (i+1):13
            % We calculate the DB index scores for all possible features
            % of the cultivars and store their score in best_pair if they
            % have the lowest one of all
            score = dbindex(Xmat(:, [i,j]), yvec);
            if score < min_score
                min_score = score;
                best_pair = [i,j];
            end
        end
    end

    % We round the DB index score of the best pair to 4 decimal places for
    % accurate interpretation and comparison later.
    min_score4 = round(min_score, 4);

    % Calculates the K-Means clusterings for all 3 cultivars using the
    % built-in K-Means function which we later use to analyse our
    % dimension-reduction results done in PCA and standardized PCA.
    [idx, ~] = kmeans(Xmat(:,best_pair), 3);

    % We plot the resulting clusters of the best pair of features for the 3
    % cultivars and its respective K-Means cluster which is an interesting
    % tool of exploration and is discussed in depth in the discussions
    % section of the report
    figure;
    tiledlayout(2,1);
    nexttile;
    
    % Plots the scatter graph showing the clusterings for the best
    % pair of features
    gscatter(Xmat(:,best_pair(1)), Xmat(:,best_pair(2)), yvec);
    title("Clustering of best pair of features (DB Index = " + num2str(min_score) + ")");
    
    % This helps us output two graphs in the same figures and makes
    % side-by-side comparisons convenient
    nexttile;
    
    % Plots the scatter graph showing the K-Means clusterings for the best
    % pair of features
    gscatter(Xmat(:,best_pair(1)), Xmat(:,best_pair(2)), idx);
    title("K-Means Clustering of best pair of features (k = " + num2str(length(unique(yvec))) + ")");
    
    % This is part (b) of the problem where we conduct the Principal
    % Component Analysis (PCA) for the data in wine.csv. We do so using the
    % right singular vectors 'V' from the Singular Value Decomposition
    % analysis of the Xmat, which contains the raw data from all the
    % features of the cultivars. The columns of V are ordered by decreasing
    % magnitude of corresponding singular values in matrix 'S', which
    % determines the order of importance based on the level of variance
    % present inside the raw data for each of the features.
    
    % We compute the mean of Xmat, which contains the raw data from all the
    % features of the cultivars, and subtract the zero-means of all
    % features from their raw data to convert Xmat to a mean-centered XmatM
    M = mean(Xmat);
    XmatM = Xmat - M;
    
    % This finds the left singular vectors 'V' through an SVD analysis on
    % the mean-centered XmatM
    [~, ~, V] = svd(XmatM);
    
    % This explicitly states the no. of principal components that we want
    % to use in our PCA, which in our case is 2. An interesting manner to
    % obtain the most efficient no. of principal components is to do 
    % r = rank(XmatM) and then plot(sum(S)/sum(sum(S))) which shows the
    % optimal no. of principal components to be used at the inflection
    % points on curve, which really is a product of the level of variance
    % in each dimension or feature which contributes most to modelling the
    % data as a whole
    compNum = 2;

    % Z2 calculates PCA scores here and obtains the principal components by
    % multiplying Xmat with V which, as outlined in the preceding comment,
    % help extract the variables that 'carry' most of the weight in term of
    % information and use them to produce a meaningful visualization
    Z2 = XmatM*V(:,1:compNum);
    
    % Calculates the DB index score for the PCA done above which is a
    % measure of evaluating the quality of clustering results obtained
    % through any decomposition or analysis. The lower the index, the
    % better the clustering
    DB_score_PCA = dbindex(Z2, yvec);
    DB_score_PCA4 = round(DB_score_PCA,4);

    % Repeats the calculation for the K-Means clusterings described in part
    % (a)
    idx_PCA = kmeans(Z2, length(unique(yvec)));

    % We plot the resulting clusters from the PCA and its respective
    % K-Means clusterings which we discuss in depth in the discussions
    % section
    figure;
    tiledlayout(2,1);
    nexttile;

    % Plots the scatter graph showing the clusterings for the PCA
    gscatter(Z2(:,1), Z2(:,2), yvec);
    title("PCA of Raw Wine Data: Reduced to 2 Dimensions (DB Index = " + num2str(DB_score_PCA) + ")");
    
    nexttile;

    % Plots the scatter graph showing the K-Means clusterings for the PCA
    gscatter(Z2(:,1), Z2(:,2), idx_PCA);
    title("K-Means Clustering of PCA-reduced Raw Wine Data (k=" + num2str(length(unique(yvec))) + ")");
    
    % This is part (c) of the problem where we conduct standardized Principal
    % Component Analysis (PCA) for the data in wine.csv. The process for
    % this is identical to the one outlined for part (b) except for the
    % salient difference where we standardize the data in Xmat. The
    % implications of this are numerous and the pros and cons of it are
    % discussed in depth in the discussions section.

    % Standardizes Xmat using the built-in function zscore
    Xmat_std = zscore(Xmat);

    % Finds the left singular vectors 'V' through an SVD analysis on
    % the standardized Xmat
    [~, ~, V2] = svd(Xmat_std);

    % The function of these statements has been outlined in part (b)
    compNum = 2;
    Z2_std = Xmat_std*V2(:,1:compNum);
    
    % Process repeated from part (b)
    DB_score_std = dbindex(Z2_std, yvec);
    DB_score_std4 = round(DB_score_std,4);
    
    % Repeats the calculation for the K-Means clusterings described in part
    % (a)
    idx_std = kmeans(Xmat_std, length(unique(yvec)));

    % We plot the resulting clusters from the standardized PCA and its respective
    % K-Means clusterings which we discuss in depth in the discussions
    % section
    figure;
    tiledlayout(2,1);
    nexttile;
    
    % Plots the scatter graph showing the clusterings for the standardized PCA
    gscatter(Z2_std(:,1), Z2_std(:,2), yvec);
    title("PCA of Standardized Wine Data: Reduced to 2 Dimensions (DB Index = " + num2str(DB_score_std) + ")");
    
    nexttile;
        
    % Plots the scatter graph showing the K-Means clusterings for the standardized PCA
    gscatter(Z2_std(:,1), Z2_std(:,2), idx_std);
    title("K-Means Clustering of PCA-reduced Standardized Wine Data (k=" + num2str(length(unique(yvec))) + ")");
    
    % Creates a table of the DB index scores calculated previously in parts
    % (a), (b) and (c) and the 2 best features that model the data best ie.
    % the 2 principal components of the data. This data summarizes the
    % calculations above and outputs them to the command window
    test_names = ["Data Columns", "Raw PCA Scores", "Standardized PCA"];
    DB_scores = [min_score4, DB_score_PCA4, DB_score_std4];
    variables = ["[" + num2str(best_pair(1)) + " "+ num2str(best_pair(2)) + "]", NaN, NaN];
    T = table(test_names', DB_scores', variables', 'VariableNames', {'Test', 'DB Index', 'Selected Variables'});
    disp(T);
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