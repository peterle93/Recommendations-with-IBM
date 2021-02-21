import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from recommender import Recommender

if __name__ == '__main__':

    rec=Recommender()

    # Quick spot check just use it to test your functions
    # normal users
    rec_names =rec.recommend_articles(20)
    print("The top 10 recommendations for user 20 are the following article names:")
    print(rec_names)
    # user with few articles.
    rec_names =rec.recommend_articles(141)
    print("The top 10 recommendations for user 141 are the following article names:")
    print(rec_names)
    # new user id 6000
    rec_names =rec.recommend_articles(6000)
    print("The top 10 recommendations for new user 6000 are the following article names:")
    print(rec_names)

    rec.draw_curve()
