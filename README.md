# Data Science Football Project: Project Overview

- Created a tool that predicts the final points total of a European football team (MAE ~ 4.2) based on input data  

- Scraped data for football teams in Europe's top leagues from Undertstat.com using python and Beautiful Soup

- Undertook exploratory data analysis to understand the key relationships and characteristics of the data

- Created and trained Linear, Lasso and Random Forest Regression Models 

- Optimised models using GridSearchCV

- Built a client facing API using Flask

## Resources

**Python Version:** 3.7

**Packages:** pandas, numpy, sklearn, matplotlib, beautifulsoup, seaborn, flask, json, pickle

**Web Framework Requirements:** See requirements.txt

**Project Adapted from:** https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

**Scraper Github:** https://github.com/slehkyi/notebooks-for-articles/blob/master/web_scrapping_understatcom_for_xG_dataset.ipynb

**Scraper Article:** https://towardsdatascience.com/web-scraping-advanced-football-statistics-11cace1d863a

**Flask Productionisation:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

Project adapted from Ken Jee's Data Science Project from Scratch Series (Glassdoor Salary Estimator) - followed along with the tutorial (link above). His channel has lots of great resources for data scientists.

## Web Scraping

Used the web scraper github repo (above) to scrape statistics for football clubs in Europe's top leagues for the 2015 - 2019 seasons. For each club we got the following:

- League
- Season Year
- Team Name
- Position
- Wins
- Draws
- Losses
- Goals Scored
- Goals Conceded
- Expected Goals (xG)
- Expected Goals Against (xGA)
- Passes Allowed per Defensive Action in Opposing Half (PPDA)
- Opponent Passes Allowed per Defensive Action in Opposing Half (OPPDA)
- Passes Complete within 20 Yards of Goal (Deep)
- Opponent Passes Complete within 20 Yards of Goal (Deep Allowed)

## EDA

I did some exploratory data analysis to better understand the relationships between different variables and summarise the main characteristics of the data. Some highlights are below. (The images can take some time to load. If they don't appear, see the images folder in the repo).

![](/Images/wins_profile_EDA.PNG)

![](/Images/pts_position_team_pivot_EDA.PNG)

![](/Images/football_heatmap_EDA.PNG)

## Model Building

First, I transformed the data structure into a dataframe and produced dummy variables where applicable. I used train_test_split to split the data into train (80%) and test (20%) sets.  

I developed three models and used GridSearchCV to optimsie the model parameters, using Mean Absolute Error (MAE) as the evaluation criteria. 

The models were:
- **Multiple Linear Regression**
- **Lasso Regression**
- **Random Forest**

## Model Performance

All of the models performed relatively well and were able to predict the final points total of the football club within +- 3 points. 
The Random Forest model was the best performing of the three models.

- **Multiple Linear Regression:** MAE = 4.19
- **Lasso Regression:** MAE = 4.23
- **Random Forest:** MAE = 4.16
 
 ## Productionisation
 
 In this stage, I made a Flask API web server following along with the TDS Tutorial and Ken Jee Data Science Project Video in the reference section above. The API takes a list of input values and returns the estimated position of the football team in the league at the end of the season.
 
 Projected values can be input into the model throughout the season to predict the final points total of the football team in the league. 
 
 



