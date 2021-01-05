# Recommendations-with-IBM

## Table of Contents
1. [Project Motivation](#motivation)
2. [Summary of Results](#results)
3. [Instructions](#instructions)
4. [Libraries Used](#libraries)
5. [File Descriptions](#descriptions)
6. [Acknowledgements](#acknowledgements)

## Project Motivation <a name="motivation"></a>
### Recommendations with IBM

For this project, we will analyze the interactions users have with articles on the IBM Watson Studio platorm and provide recommendations on which new articles you think they will like. To determine which articles to show to each user, we will be performing a study of the data available on the IBM Watson Studio platform. 

The project will be divided into the following parts:

### I. Exploratory Data Analysis

Before making recommendations, we will did to explore the data that we are working with for the project. We'll answer the basic required questions in the data that that we'll be working on throughout the rest of the notebook before we dive into the details of the recommendation system in the later sections.

### II. Rank Based Recommendations

To begin building recommendations, we'll need to find the most popular articales simply based on the most interactions. Since there are no ratings for the articles, we can easily assume the articles with the most interactions are the most popular. These will be the articles we might recommend to new suers (or anyone depending on what we know about them.

### III. User-User Based Collaborative Filtering

To build better recommendations for the users of IBM's platform, we could look at users that are similar in terms of the items they have interacted with. The items could then be recommended to the similar users. This would be a step in the right direction towards more personal recommendations for the users. This step will be implemented next.

### IV. Matrix Factorization

We will complete machine learning approach to building recommendations. The user-item intereactions will build out a matrix decomposition where we'll gain an idea of how well we can predict new articles an individual might intereact with. We will also discuss which methods we may use moving forward and how well our recommendations are working for engaging users.

## Summary of Results <a name="results"></a>


## Instructions: <a name="instructions"></a>


## Libraries and Dependencies: <a name="libraries"></a>

Python 3.6.6+

1. pandas
2. numpy
3. matplotlib
4. pickle 
5. scikit-learn

## File Descriptions <a name="descriptions"></a>

## Acknowledgements <a name="acknowledgements"></a>
1. https://www.ibm.com/ca-en
2. https://www.udacity.com/
