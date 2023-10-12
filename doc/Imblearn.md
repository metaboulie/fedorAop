# Imblearn

## Steps

1. Preprocess all the datasets
   - For each train set, fit a [`PCA()`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)  and a [`MinMaxScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler) to reduce dimensionality and scale the data
   - For each corresponding test set, transform the data with fitted `PCA()`  and a `MinMaxScaler()`
   - Make the following sampling converge faster
2. Train the model
   - Iterate over all samplers from[ `Imbalanced learn`](https://imbalanced-learn.org/stable/user_guide.html#user-guide)
   - For each sampler
     - Iterate over the all datasets
     - In each dataset
       - Utilize this sampler to sample the train data
       - Fit a [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) with the sampled data
       - Predict the whole test data with the classifier
       - Record the [`geometric_mean_score`](https://imbalanced-learn.org/stable/references/generated/imblearn.metrics.geometric_mean_score.html)

## Evaluation

1. For a sampler, record its `geometric_mean_score` for all datasets respectively
2. After all samplers are processed, rank the `geometric_mean_score` of them in each dataset, the highest has rank 1, the second has rank 2, and so on
3. Calculate the average rank for each sampler
4. Output the results
   - The 4 highest `geometric_mean_score` for every dataset are highlighted
   - The 4 samplers with highest average rank are highlighted

![imblearn_results](../images/imblearn_results.png)

## Summary

1. The `RandomUnderSampler`, `ClusterCentroids`, `SMOTEENN`, `SVMSMOTE` get the highest average rank