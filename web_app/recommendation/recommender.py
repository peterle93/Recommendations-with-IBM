import pandas as pd
import numpy as np
# web app use plotly , so remove : import matplotlib.pyplot as plt
from recommendation.data_clean import Data_Clean
from recommendation.rbrecommender import RBRecommender
from recommendation.ucfrecommender import UCFRecommender
from recommendation.mfrecommender import MFRecommender
from recommendation.cbrecommender import CBRecommender

class Recommender():
    '''
    Class: User Based Collaborative Filtering Recommendations
    '''
    def __init__(self, interact_pth='data/user-item-interactions.csv', articles_pth='data/articles_community.csv',top_n=10):
        self.top_n = top_n
        dc=Data_Clean(interact_pth, articles_pth)
        self.ucfr=UCFRecommender(dc.interacts_clean)
        self.rbr=RBRecommender(dc.interacts_clean)
        self.cbr=CBRecommender(dc.articles_clean,dc.interacts_clean)
        self.mfr=MFRecommender(dc.articles_clean,dc.interacts_clean)
        self.user_with_few_articles = self.ucfr.user_item.index[self.ucfr.user_item.sum(axis=1)<3].values

    def recommend_articles(self, user_id, top_n=10):
        '''
        Description:
         Acording to users type:
           new user: RBRecommender
           old user and articles : UCFRecommender/MFRecommender
           user reading few articles: CBRecommender
        Args:
          user_id
        Return:
          Recs: list of recommendations
        '''
        if top_n != 10:
            self.top_n = top_n
        if user_id in self.cbr.users:
            if user_id in self.user_with_few_articles:
                recs=self.cbr.make_content_recs(user_id, self.top_n)
                if len(recs) == 0:
                    recs=self.rbr.get_top_articles()
            else:
                _, recs=self.ucfr.user_advance_recs(user_id, self.top_n)
        else:
            recs=self.rbr.get_top_articles()
        return recs

    def mf_calculate_error(self):
        return self.mfr.calculate_error()
    '''
    # Web server use ploly to draw curve, so comment out the following.
    def draw_curve(self):
        latent_factors_num,test_accuracy,train_accuracy =self.mfr.calculate_error()
        self.mfr.draw_curve(latent_factors_num,test_accuracy,train_accuracy)
    '''
if __name__ == '__main__':

    rec=Recommender()

    # Quick spot check just use it to test your functions
    # normal users
    rec_names =rec.recommend_articles(20)
    print("The top 10 recommendations for user 20 are the following article names:")
    print(rec_names)
    # user with few articles.
    rec_names =rec.recommend_articles(141)
    print("The top 10 recommendations for user 2 are the following article names:")
    print(rec_names)
    # new user id 6000
    rec_names =rec.recommend_articles(6000)
    print("The top 10 recommendations for new user 6000 are the following article names:")
    print(rec_names)

    #rec.draw_curve()
