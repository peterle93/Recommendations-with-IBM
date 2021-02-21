import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
# Import linear_kernel to compute the dot product
from sklearn.metrics.pairwise import linear_kernel
class CBRecommender():
    '''
    Class: Content Based Recommendations
    You might consider content to be the doc_body, doc_description, or doc_full_name. There isn't one way to create a content based recommendation,
    especially considering that each of these columns hold content related information.

    1. Use the  below to create a content based recommender. Since there isn't one right answer for this recommendation tactic,
    no test functions are provided. Feel free to change the function inputs if you decide you want to try a method that requires more input values.
    The input values are currently set with one idea in mind that you may use to make content based recommendations.
    One additional idea is that you might want to choose the most popular recommendations that meet your 'content criteria',
    but again, there is a lot of flexibility in how you might make these recommendations.
    '''
    def __init__(self, articles_clean,interacts_clean,top_n=10):
        self.top_n = top_n
        self.df_content=articles_clean
        self.df = interacts_clean
        self.content_cosine=self.create_content_cosine_similar()
        # prepare the article location and user reading datafram
        self.users=self.df.user_id.unique()
        self.reading_users=interacts_clean
        self.reading_users=self.reading_users.reset_index().drop('index',axis=1)
        self.reading_users.set_index('user_id',inplace=True)

    def create_content_cosine_similar(self):
        '''
        Args:
        Return:
          content_cosine_sim: np, article_id by article_id
        '''
        #Define a TF-IDF Vectorizer Object. Remove all english stopwords
        tfidf = TfidfVectorizer(stop_words='english')
        #Replace NaN with an empty string
        df_articles=self.df_content['doc_description']
        #Construct the required TF-IDF matrix by applying the fit_transform method on the doc_description feature
        tfidf_articles = tfidf.fit_transform(df_articles)
        # Compute the cosine similarity matrix
        content_cosine_sim = linear_kernel(tfidf_articles, tfidf_articles)
        return content_cosine_sim

    def find_similar_article_ids(self,article_id):
        '''
        INPUT
        article_id -string: a article_id
        OUTPUT
        similar_article_ids - an array of the most similar article id
        '''
        try:
            # find the row of each article id
            article_idx =  np.where(self.df_content['article_id'].astype(float).astype('str') ==  article_id)[0][0]

            # find the most similar article indices - to start I said they need to be the same for all content
            max_similar_idx=np.argsort(self.content_cosine[article_idx])[-2:-5:-1] # -1 is current article id, -2 is max similar usrs.

            # pull the article ids based on the similar_idxs
            raw_article_ids=self.df_content.iloc[max_similar_idx,4].values.astype('float').astype('str')

            # filter out article id doesn't exist in artical content dataset.
            similar_article_ids=np.array([aid for aid in raw_article_ids if aid not in self.article_id_without_content()])
        except :
            #if df_content.article_id doesn't include the df.article_id, keyError show, return empty np.
            similar_article_ids=np.array([])

        return similar_article_ids
    def get_article_names(self, article_ids):
        '''
        INPUT:
        article_ids - (list) a list of article ids
        self.df - (pandas dataframe) df as defined at the top of the notebook

        OUTPUT:
        article_names - (list) a list of article names associated with the list of article ids
                        (this is identified by the title column)
        '''
        article_names=self.df.drop_duplicates().loc[self.df['article_id'].isin(article_ids),'title'].unique().tolist()
        return article_names # Return the article names associated with list of article ids

    def article_id_without_content(self):
        '''
        Description: Get the article id that doesn't exist in article content datasets
        '''
        df_rid=self.df['article_id'].astype('str').unique()
        df_crid=self.df_content['article_id'].astype('float').astype('str').unique()
        return np.setdiff1d(df_rid, df_crid).astype('str')

    def make_content_recs(self, user_id, top_n=10 ):
        '''
        INPUT
        user_id: a user id
        reading_users: dataframe which removes duplicated rows and includes user id, article id and title
        None
        OUTPUT
        recs
        '''
        recs_list = []
        if top_n != 10:
            self.top_n = top_n
        import pdb
        #pdb.set_trace()
        # Pull only the reviews the user has seen
        if user_id not in self.reading_users.index.values:
            return ("Content based recommendation cannot work for new user.")
        users_temp = self.reading_users.loc[user_id] # users_temp is user list that read articles.
        # artcle_ids  is article ids np that users read.
        article_ids = np.array(users_temp.drop_duplicates()['article_id'].astype('str'))
        if  article_ids.size == 1:
            article_ids=[article_ids.tolist()]

        # Look at each of the articles (most similar first),
        # pull the articles the user hasn't seen that are most similar
        # These will be the recommendations - continue until 10 recs
        # or you have depleted the article list for the user
        for art_id in article_ids:
            rec_art_ids = self.find_similar_article_ids(art_id)
            #remove article_id that current user read.
            temp_rec_ids = np.setdiff1d(rec_art_ids, article_ids)
            temp_recs = self.get_article_names(temp_rec_ids)
            if len(temp_recs) > 0:
                recs_list +=temp_recs

            # If there are more than
            if len(recs_list) > self.top_n:
                recs_list=recs_list[:self.top_n]
                break

        return recs_list

    def make_recs(self,users=None):
        '''
        INPUT
        users: user id list
        reading_users: dataframe which removes duplicated rows and includes user id, article id and title
        None
        OUTPUT
        recs - a dictionary with keys of the user and values of the recommendations
        '''
        # Create dictionary to return with users and ratings
        recs = defaultdict(list)
        recs_list=[]
        if users is None:
            users=self.users
        # For each user
        for user in users:
            recs_list=self.make_content_recs(user)
            recs[user]=recs_list

        return recs


if __name__ == '__main__':
    from data_clean import Data_Clean
    from cbrecommender import CBRecommender

    #instantiate data clean
    dc = Data_Clean()
    cbr=CBRecommender(dc.articles_clean,dc.interacts_clean)
    results=cbr.make_recs(users=[3,141,147])
    for i, arts in results.items():
        for k, a in enumerate(arts):
            print("Recommend user {} to read article {}:{}".format(i,k+1, a))
