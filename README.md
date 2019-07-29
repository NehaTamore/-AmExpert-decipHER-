# -AmExpert-decipHER-
My solution for American Express and Analytics Vidhya presented “AmExpert decipHER – Hackathon, predicting credit card spend using advanced regression techniques.

This was my first attempt and in the direction not to just score better on the leaderboard with many many ensembles at the cost of complexity but to generate an elegant, neat and simple model to predict the target variable, considering the trade off between resources and the performance metric, as in production systems. :)
On side note this resulted in rank 37 on the private leaderboard and the score was pretty much consistent with that of public! Overfitting is a no no!!! :P

Quick overview of my approach and analysis

Major deciding factors in the competition: 
0. Featurizations
1. Imputing missing values
2. Modeling
3. Error analysis
4. Feature transformations with error analysis

0. Featurizations
- Considering the plots and univariate analysis of every feature there were many of them were skewed. Thus for every feature with skewedness greater than a threshold, log1p transform of that feature is taken. 
- The 2 features which had smoother curve, on sqrt transform were changed accordingly.
- Label encoding of categorical data was done. Region_code could have been one-hot encoded, But the sparse feature vector didn't carry as much information. Thus simple normalized value of counts of region code was formulated

Things to try:
- Embedding for region code
- Taking expected value of target variable, by grouping target by region_code
- Run a small NN on one-hot encoded region codes to predict the target. and use the weights and prediction as features. Similar to embedding.


1. Imputing missing values approaches
- Mean
- Median
- zero valued
- using imputer classes with round robin technique
When 90% or more values were missing for a binary variable and rest were all 1, it was hypothised that, the missing value meant absence of that variable. For example personal_loan had 97% missing values and 3% values were 1s. It was safe to assume that for the 97% cases the personal_loan was not taken. This approach worked far better than filling in values with mean or median.
Apart from that, it reduced the unnecessary computational overhead

For values which were numerical and about 50% of missing data, mean or median of the feature column was considered to fill in.

Features such as investment_4 had 42 negative values, these outliers changed to median of the feature value

using imputer classes with round robin technique provided marginal reduction in cv score, but trade of complexity vs gain didn't pann out well

2. Modeling
- Ensemble models of GBDT, RF, ExtraTrees was used, which all performed fairly similarly
- Ridge lasso regression techniques with above features did not work, as well, as could be noticed that target was not linearly correlated with any of the depended variables, which is the primary assumption of linear models.
- Experiments with interaction polynomial features of degree 2 with linear models(tried even with ensembles)resulted in no significant improvement.
- GridSearch for hyperparameters was performed. Considering Overfitting the public test data, it wasn't done rigorously 

3. Error analysis
- The residual plots, true vs pred plots, the joint and histogram distribution of true and false for each model were examined against their r2 score, rmse.
Most of the residual plots for all ensembles looked similar, with randomly scattered points. For linear models, the residual plots did have a pattern. And various feature transformations were performed to improve the plot, against r2 score.
- For ensemble models, the r2 and residual plots were fairly satisfactory and thus moved on to error analysis

4. Error analysis. A different bit added
- For every model, the cv data was concatenated with residues, absolute value of residues and it was sorted in descending order to see what points our model are incorrect about. Or what pattern causes most error.
- Clustering analysis on this data could provide the subset of datapoints which our model gets wrong. And manually tweeking points could make us understand the error better. (To do!)

5. Based on some plots, sqrt transform on 2 features gave marginal improvement in the cv score!
- And some clipping of the predicted scores was explored. (if model predicts too high or too low values based on distribution of target in train set, these values were clipped)
- Another interesting thing was to analyse the true vs pred scatter plot to comprehend that model slightly predicts lower values than the true.( the scatter graph was slightly shifted below x=y line) Manually adding thresholds of values to target variable was explored!

Something to try: 
In data dictionary we have 9 variables, corresponding to information about debit or credit card finances of each month, april may and june, which have sequential nature. And rest of the 17 variables, like gender, region code are non-sequential. This problem could be modelled as below with either very small RNN(number of hidden units), cause we have small dataset with around 44 features, so we don't want RNNs to literally remember every training sample. Had it been larger dataset this architecture, could be one of the most appropriate architectures to model this sequential nature in the problem!


![rnn architecture](https://github.com/NehaTamore/-AmExpert-decipHER-/blob/master/Untitled%20Diagram%20(2).jpg)

