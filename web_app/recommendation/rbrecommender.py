import pandas as pd
import numpy as np

class RBRecommender():
    '''
    Class: Rank-Based Recommendations
    we don't actually have ratings for whether a user liked an article or not. We only know that a user has interacted with an article.
    In these cases, the popularity of an article can really only be based on how often an article was interacted with.
    '''
    def __init__(self, df,top_n=10):
        self.top_n = top_n
        self.df=df

    def get_top_articles(self, n=10):
        '''
        INPUT:
        n - (int) the number of top articles to return
        df - (pandas dataframe) df as defined at the top of the notebook

        OUTPUT:
        top_articles - (list) A list of the top 'n' article titles

        '''
        if n != 10:
            self.top_n = n
        df_tmp=self.df.groupby(['article_id','title']).size()
        df_article=df_tmp.sort_values(ascending=False).head(self.top_n).reset_index()
        top_articles=df_article['title'].values.tolist()
        return top_articles # Return the top article titles from df (not df_content)

    def get_top_article_ids(self, n=10):
        '''
        INPUT:
        n - (int) the number of top articles to return
        df - (pandas dataframe) df as defined at the top of the notebook

        OUTPUT:
        top_article_ids - (list) A list of the top 'n' article ids
        '''
        if n != 10:
            self.top_n = n
        df_article=self.df.groupby(['article_id','title']).size().nlargest(self.top_n).reset_index() #
        top_article_ids=df_article['article_id'].values.astype('str').tolist()
        return top_article_ids # Return the top article ids

if __name__ == '__main__':
    from data_clean import Data_Clean
    from rbrecommender import RBRecommender

    #instantiate data clean
    dc = Data_Clean()
    print(dc.articles_clean.shape, dc.interacts_clean.shape)

    rbr=RBRecommender(dc.interacts_clean)
    print(rbr.get_top_articles())
    print(rbr.get_top_article_ids())
