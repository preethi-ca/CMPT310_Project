# CMPT310_Project
#### Full names & SFU emails of members: 
Romandeep Singh Chauhan (rsc18@sfu.ca), Shelby Haines (srh11@sfu.ca), Preethi Chidambaram (pca115@sfu.ca), Aymen Jerbi (aja176@sfu.ca)
## Description
The goal of our project is to predict the success of a new restaurant/cafe location using supervised machine learning techniques like regression and classification on demographic, geographic, and competitive environment inputs (specific inputs mentioned below). Our goal is to predict a continuous rating (target_rating) or a discrete success threshold (target_is_successful), which we will evaluate using MSE, R-squared, and F1-score metrics.


## Dependencies
 - [Pandas](https://pandas.pydata.org/docs/)
 - [NumPy](https://numpy.org/)
 - [Scikit Learn](https://scikit-learn.org/0.21/documentation.html)
 - [XGBoost](https://xgboost.readthedocs.io/en/stable/)
 - [Requests](https://requests.readthedocs.io/en/latest/)
 - [GeoPandas](https://geopandas.org/en/stable/docs.html)
 - [Pycancensus]()

On macOS, XGBoost may require the OpenMP runtime (`brew install libomp`).
 

*Right now the project is being worked on in multiple different branches (not all of it is in the main branch), so to view all progress please look in all branches.*

The goal of this project is to predict the success of a new restaurant/cafe location. The model works by by taking information about the demographic, geographic, and competitive environment as inputs, including:
- `median_income`, `pop_density_sqkm`, `age_demographic`, `competitor_count_500m`, `transit_distance_m`, `store_name`, `city`, `latitude`, `longitude`, `target_rating`, `target_is_succesful`

Model scripts currently include KNN and XGBoost classification workflows that reuse the same engineered feature set.

### KNN vs XGBoost Results
Using the same engineered features and stratified train/validation/test splits:

| Model | Holdout Accuracy | Holdout F1 | Takeaway |
| --- | ---: | ---: | --- |
| KNN | ~0.69 | ~0.81 | Slightly better final holdout score |
| XGBoost | ~0.67 | ~0.80 | Comparable performance, more room for tree-based tuning |
