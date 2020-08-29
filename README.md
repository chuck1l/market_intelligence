# <div align="center">_**Forecasting Action With Machine Learning/Artificial Intelligence**_</div>
## <div align="center">_**Lawrence C. Williams - The Savvy Trading Machine**_</div>

<img src="https://github.com/chuck1l/market_intelligence/blob/master/data/header.png" height="70" width=100% />

## _**Introduction:**_

An effective trading strategy is essential in attempting to sustain a living as a retail trader. It is without a doubt challenging, but exciting at the same time. The most consistent way, to avoid becoming an all too familiar statistical loser in the stock market, includes a necessary combination of advanced strategy, risk management, discipline, money management, emotional intelligence and now the proper computational tools utilizing machine intelligence. In the end, the trader's success depends on the proven, straight-forward strategy that is flexible enough to be effective for any security that meets our criteria.

No matter which strategy is chosen, the success or failure truly depends on three main components that can't be ignored:

* **Volume:** The volume informs us as to how much the instrument has been traded in the focal period. Elevated volume speaks to the significant change in interest for the asset or security. Usually (EDA focus), an increase in this metric will educate the informed trader that an impending price jump is approaching, price movement up or down doesn't matter, we can capitalize.

* **Volatility:** You can forecast the potential profit range based on historical volatility. Assuming that that price movement will recur in a manner consistent with past events - following support and resistance points, and trend lines can help predict the action.  However, elevated volatility will not only impact your profit potential, the traders risk potential is greater at the same time.  Whiplash or reversals can eat away profits very fast in a volatile market.

and last, but certainly not least:

* **Liquidity:** Liquidity enables the quick movement in and out of a position.  A trader's edge often depends on small price deltas, a lot of that can be lost if a large spread exists between the bid and ask.  This happens from a lack of liquidity (The Law of Supply and Demand). Profits erode quickly if an asset or security is illiquid, preventing the trader from executing at the target price.  Settling for the much lower bid, or much higher ask!

Before we submerge ourselves in the complexity that is the world of highly technical indicators, our exploratory analysis should focus on the basics. Often people think the basics aren't enough to improve upon the win-ratio. I am developing this introduction to argue that we will accomplish just that with the incorporation and analysis of principal elements. We will let exploratory data analysis and automated processes do the heavy lifting in the technicals, allowing more time to focus on discipline and emotional intelligence for the win.

## _**The Pipeline:**_

1. Selecting the security from over-night action (Gaps)
  * Looking for securities that have gained attention from other traders/algorithms with acceptable ~3% (+ or -) gap
  * Preferred market float and relative volume to ensure volatility and liquidity 
  
2. Perform web scraping on the news associated with the selected financial instrument
  * "news" source: recent blogs, chats, social media, surveys, articles, and documents
  * Article cleaning - remove punctuation and words that provide no value, lemmatize and vectorize content
  * Topic modeling and text analysis - Latent Dirichlet Allocation (LDA) and/or Non-Negative Matrix Factorization (NMF)
  * Sentiment analysis on the topics and text
  * Perform VADER sentiment analysis to forecast directional movement out of the gate (opening bell)
      * VADER (Valence Aware Dictionary and sEntiment Reasoner): a lexicon and rule-based analysis tool specifically attuned to sentiments expressed
      
3. Import historical data over a timeframe that provides the most effective forecasting for today's action
  * Price information and technical indicators
  * Perform Exploratory Data Analysis (EDA)
  * Normalization of data (scale all variables to have a values between 0 and 1)
  * Dimensionality Reduction to remove all features that produce multicollinearity or provide no value in the modeling
      * Principal component analysis (PCA)
      
4. Classification modeling for multiple purposes
  * Binary Classification utilized to predict directional movement (up or down) out of the gate (opening bell)
      * Random Forests Classifier, Recurrent Neural Networks Classifier, and/or Support Vector Machines (SVMs)
  * Multi-Class Classification utilized to predict zones (time of day) for expected "high of day" and "low of day"
      * Random Forests Classifier, Recurrent Neural Networks Classifier, and/or Support Vector Machines (SVMs)
      
5. Time Series Analysis, Forecasting the price zones for the upcoming day's "high of day" and "low of day" marks
  * Keras and TensorFlow
      * Long Short-Term Memory (LSTM)
  * Gradient Boosting Regressor
      * Randomized Search CV for best parameters
      * Grid Search CV for further improved parameters
  * Random Forest Regressor
      * Randomized Search CV for best parameters
      * Grid Search CV for further improved parameters  

## _**High-Level Explanation:**_

1. Select the instrument deemed worthy of attention.

2. Look over the internet for recent news in regards to the instrument, search for sentiment.

3. Import the data to analyze and predict on. Clean it, prepare it and remove the data the doesn't impact our signal.

4. Predict if the price movement will be up or down at the bell, and at which times to expect our entries and final exits.

5. Forecast the expected price marks (zones) for said entry and final exit, and highest probability profit points within.

6. Display the forecasted values on our website for members only to see, and chat with them during the trading day.

7. Update the results in real-time (delays for computation) as the day proceeds.
  
## _**Conclusion:**_ 

This is my current plan of action, most of the substance is subject to change as I move forward in the analysis. Realizing the best or better methods, and reducing the complexity of the algorithm. As you noticed there are multiple methods within each step of the various accomplishments. They can be minimized or reduced to single models as we visualize error measurement comparisons that lead to the most accurate method(s). If consistently better in predictions. Otherwise, they can all stay and the most effective model will be selected in real-time based on which provides the most accurate results on any given instrument. Adding computational expense and time which might hinder the speed of results during the intraday projections.

The data is always split into "Training Data" to train the model, "Testing Data" to test the model, and "Holdout Data" to further test on data that has not been seen in any way by the selected model. This is important to prevent data leakage that might mislead us in the final result.

## _**Results From My Preliminary Analysis:**_

I have already been developing some test modeling on SPY for the regressors only. Working on predicting tomorrow's values without yet including the two classification steps of "up or down at the bell" or the "Time of high and low of day prices" mentioned above. See the images below for something I consider more of the rudimentary analysis and hyper-parameter selection.

### _**Important Notes**_

y_hat_high (or low) = the predicted value from the model
true_high (or low) = the actual price of that day

<p align="center">
<img src="https://github.com/chuck1l/market_intelligence/blob/master/imgs/high_of_day.png" height="400" width=45% /><img src="https://github.com/chuck1l/market_intelligence/blob/master/imgs/low_of_day.png" height="400" width=45% />
</p>

  
  
  
  
  

