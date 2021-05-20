June 1, 2021

<dl>
<dt>Project Team</dt>
<dd>Nicholas Miller* (nmill@umich.edu)</dd>
<dd>Tim Chen* (ttcchen@umich.edu)</dd>
<dt>Github Repository</dt>
<dd><a href='https://github.com/cm-tle-pred/tle-prediction'>https://github.com/cm-tle-pred/tle-prediction</a></dd>
</dl>

\* equal contribution

# Table of Contents
<!-- TOC depthFrom:1 depthTo:3 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Supervised Learning](#supervised-learning)
	- [Supervised Learning Methods](#supervised-learning-methods)
	- [Evaluation](#evaluation)
	- [Failure Analysis](#failure-analysis)
- [Unsupevised Learning](#unsupevised-learning)
	- [Motivation](#motivation)
	- [Data Source](#data-source)
	- [Unsupervised Learning Methods](#unsupervised-learning-methods)
	- [Unsupervised Evaluation](#unsupervised-evaluation)
- [Discussion](#discussion)
- [Statement of Work](#statement-of-work)
- [Appendix A](#appendix-a)

<!-- /TOC -->


# Introduction
*Satellite positions can be calculated using publically available TLE (two-line element set) data.  This standardized format has been used since the 1970’s and can be used in conjunction with the SGP4 orbit model for satellite state propagation.  Due to many reasons, the accuracy of these propagations deteriorates when propagated beyond a few days.  Our project aims to create better positional and velocity predictions which can lead to better maneuver and collision detection.*

To accomplish this, we created a machine learning pipeline that takes in a TLE dataset and is split it into train, validate and test sets. The training set is then sent to an unsupervised model for anomaly detection outliers are removed and then feature engineering is performed.  Finally, supervised learning models are trained and tuned based on the validation set.

# Supervised Learning

## Supervised Learning Methods
> * Briefly describe the workflow of your source code, the learning methods you used, and the feature representations you chose.
> * How did you tune parameters?
> * What challenges did you encounter and how did you solve them?

*We plan to start with a Linear Regression and a simple NN with a single layer using a sample of the dataset as baseline models before moving on to a deep neural network (DNN).  Our data will consist of a normalized set of TLE variables with a target epoch as our input variables and the normalized TLE variables at the target epoch as the output variables. To account for natural effects that impact orbital mechanics, we will combine additional datasets on climate change, global temperature, solar cycles, and solar sunspot to improve accuracy.*

## Evaluation
> * Provide a correct and comprehensive evaluation, analyzing the effectiveness of both your methods, and your feature representations. Methods include e.g. ablation tests to identify important features, hyperparameter sensitivity, and training data curves. This should be done for each learning framework and representation choice, together with a valid and effective comparison between the chosen approaches.
> * Which features were most important and why?
> * What important tradeoffs can you identify?
> * How sensitive are your results to choice of parameters, features, or other varying solution elements?

*Mean square error on the baseline models’ predictions will be used as benchmarks for evaluation.  We will also consider using a custom loss function on predicted spatial x, y, z positions.  To visualize the model's effectiveness, we can plot propagated satellite positions using the SGP4, our baseline models, the DNN model against the true dataset (other TLEs) together in 3D.  Visualizing the errors for the models based on individual feature variances will also show where the strengths and weaknesses of the models lie.*

## Failure Analysis
> * Select three examples where prediction failed, and analyse why. You should be able to find at least three different types of failure.


# Unsupevised Learning
*Due to the impact of bad TLE data on the supervised learning models, we want to use unsupervised learning to remove these abnormal data from the dataset before training the data.*

## Motivation
> 1. Briefly state the nature of your work and why you chose it.
> 2. What specific question, goal, or task did you try to address related to structure in the data (e.g. the clusters you found)?

## Data Source
> Describe the properties of the dataset (or data API service) you used. Be specific. Your i nformation at a minimum should include but not be l imited to:
>
> * where the datasets or API resource i s l ocated,
* what formats they returned/used,
* what were the i mportant variables contained i n them,
* how many records you used or retrieved (if using an API), and
* what time periods they covered (if there i s a time element)
>
> For example, if you downloaded data or used API services, you should state the specific URLs to those files or resources in a way that is trivial for the instructor to retrieve them if needed.

## Unsupervised Learning Methods
> * Briefly describe the workflow of your source code, the learning methods you used, and the feature representations you chose.
> * How did you tune parameters?
> * What challenges did you encounter and how did you solve them?

*Anticipated data manipulation includes normalizing features, handling cyclic data, and treating variance in time series intervals.  Since the data anomaly usually applies to all features in a single data point, we will be trying out LocalOutlierFactor and DBSCAN from scikit-learn on highly predictive TLE features such as eccentricity and inclination.*

## Unsupervised Evaluation
> * What i nteresting relationships or insights did you get from your analysis?
> * What didn't work, and why?
> * To summarize your findings, i nclude at l east two visualizations (chart, plot, tag cloud, map or other graphic) that summarize your analysis.

*To evaluate our results' quality, we will manually check them against bad data that have been identified previously.  A visualization that will aid the evaluation process is to plot the TLE data out while highlighting the outliers that we removed.  Another exciting visualization that we will create is to plot the frequency of these data points based on when they occurred to see if we can identify additional insights or patterns.*

# Discussion
> * What did you l earn from doing Part A? What surprised you about your results? How could you extend your solution with more time/resources?
> * What did you l earn from doing Part B? What surprised you about your results? How could you extend your solution with more time/resources?
> * What ethical i ssues could arise i n providing a solution to Part A, and how could you address them?
> * What ethical i ssues could arise i n providing a solution to Part B, and how could you address them?

# Statement of Work

<dl>
<dt>Nicholas Miller</dt>
<dd>Data split strategy</dd>
<dd>Feature Engineering</dd>
<dd>Initial neural network models</dd>
<dd>Deep learning ResNet28 model</dd>
<dd>Training and Evaluation</dd>
<dd>Final Report</dd>
<dt>Tim Chen</dt>
<dd>Data collection</dd>
<dd>Unsupervised anomaly detection model</dd>
<dd>Deep learning bilinear model</dd>
<dd>Training and Evaluation</dd>
<dd>Final Report</dd>
</dl>

# Appendix A
