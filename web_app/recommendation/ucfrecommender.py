import pandas as pd
import numpy as np

class UCFRecommender():
    '''
    Class: User Based Collaborative Filtering Recommendations
    '''
    def __init__(self, df, top_n=10):
        self.top_n = top_n
        self.df=df
        self.user_item=self.create_user_item_matrix()

    '''
    1. Use the function below to reformat the df dataframe to be shaped with users as the rows and articles as the columns.
    Each user should only appear in each row once.
    Each article should only show up in one column.
    If a user has interacted with an article, then place a 1 where the user-row meets for that article-column. It does not matter how many times a user has interacted with the article, all entries where a user has interacted with an article should be a 1.
    If a user has not interacted with an item, then place a zero where the user-row meets for that article-column.
    Use the tests to make sure the basic structure of your matrix matches what is expected by the solution.
    '''
    def create_user_item_matrix(self):
        '''
        Description:
        create the user-article matrix with 1's and 0's.
        Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with
        an article and a 0 otherwise
        INPUT:
        self.df - pandas dataframe with article_id, title, user_id columns
        OUTPUT:
        self.user_item - user item matrix
        '''
        # Fill in the function here
        # drop duplicate row and count interact
        user_item=self.df.drop_duplicates().groupby(['user_id','article_id']).size().unstack(level=1)
        # fill Null to 0
        user_item=user_item.fillna(0)
        # conver float to int
        user_item=user_item.astype('int')
        return user_item # return the user_item matrix

    def find_similar_users(self, user_id):
        '''
        Description:
        Computes the similarity of every pair of users based on the dot product
        Returns an ordered

        Args:
        user_id - (int) a user_id
        self.user_item - (pandas dataframe) matrix of users by articles:
                    1's when a user has interacted with an article, 0 otherwise

        Return:
        similar_users - (list) an ordered list where the closest users (largest dot product users)
                        are listed first
        '''
        # compute similarity of each user to the provided user
        user_sim=self.user_item.loc[user_id,:].dot(self.user_item.T)
        # sort by similarity
        user_sim=user_sim.sort_values(ascending=False)

        # remove the own user's id and return user list
        most_similar_users=user_sim.loc[~(user_sim.index==user_id)].index.values.tolist()
        return most_similar_users # return a list of the users in order from most to least similar

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


    def get_user_articles(self, user_id):
        '''
        INPUT:
        user_id - (int) a user id
        self.user_item - (pandas dataframe) matrix of users by articles:
                    1's when a user has interacted with an article, 0 otherwise

        OUTPUT:
        article_ids - (list) a list of the article ids seen by the user
        article_names - (list) a list of article names associated with the list of article ids
                        (this is identified by the doc_full_name column in df_content)

        Description:
        Provides a list of the article_ids and article titles that have been seen by a user
        '''
        # Your code here
        a_user_items=self.user_item.loc[user_id,:]
        article_ids=a_user_items[a_user_items==1].index.values.astype('str').tolist()
        article_names=self.get_article_names(article_ids)
        return article_ids, article_names # return the ids and names


    def user_recs(self, user_id, top_n=10):
        '''
        Description:
        Loops through the users based on closeness to the input user_id
        For each user - finds articles the user hasn't seen before and provides them as recs
        Does this until m recommendations are found
        Notes:
        Users who are the same closeness are chosen arbitrarily as the 'next' user
        For the user where the number of recommended articles starts below m
        and ends exceeding m, the last items are chosen arbitrarily

        Args:
        user_id - (int) a user id
        top_n - (int) the number of recommendations you want for the user

        Return:
        recs - (list) a list of recommendations for the user
        '''
        if top_n != 10:
            self.top_n = top_n

        # define variable:
        closeness_user_ids=[]
        current_user_reading_ids=[]
        closeness_user_reading_ids=[]
        rec_aids=[]

        # According to user_id to find out the most similar user_ids list(closeness_user_ids)
        closeness_user_ids=self.find_similar_users(user_id)
        # Get the current user's reading  ids list(current_user_reading_ids)
        current_user_reading_ids,_=self.get_user_articles(user_id)
        # For each user - finds articles the user hasn't seen before and provides them as recs
        for u_id in closeness_user_ids:
            ## Get the reading id lists(closeness_reading_ids) of each user in the closeness_user_reading_ids
            closeness_user_reading_ids,_=self.get_user_articles(u_id)
            ## add into rec_aids if closeness_user_reading_ids not in current_user_reading_ids and top_n < 10
            rec_aids +=list(np.setdiff1d(closeness_user_reading_ids,current_user_reading_ids))
            if len(rec_aids) > self.top_n:
                break
        recs=rec_aids[:self.top_n]
        return recs # return your recommendations for this user_id

    def get_top_sorted_articles(self, article_ids):
        '''
        Description:
             Choose articles with the articles with the most total interactions
             before choosing those with fewer total interactions.
        Args:
        article_ids - (str) article id list
        user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

        Return:
        recs - (list) a list of recommendations for the user by article id
        rec_names - (list) a list of recommendations for the user by article title

        Other Details - sort the df_articles by number of interactions where highest of each is higher in the dataframe
        '''
        #sort the df_articles by number of interactions where highest of each is higher in the dataframe
        df_articles=self.user_item.sum(axis=0).sort_values(ascending=False).reset_index()
        df_articles.columns=['article_id','num_interactions']

        # get the ordered article_ids sort by num_interactions
        sorted_aids=df_articles.loc[df_articles['article_id'].isin(article_ids),'article_id'].values.tolist()
        sorted_names=self.get_article_names(sorted_aids)
        return sorted_aids,sorted_names


    def get_top_sorted_users(self, user_id):
        '''
        INPUT:
        user_id - (int)
        self.user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise


        OUTPUT:
        neighbors_df - (pandas dataframe) a dataframe with:
                        neighbor_id - is a neighbor user_id
                        similarity - measure of the similarity of each user to the provided user_id
                        num_interactions - the number of articles viewed by the user - if a u

        Other Details - sort the neighbors_df by the similarity and then by number of interactions where
                        highest of each is higher in the dataframe

        '''
        # according to user_id to get most similar neighbor_ids and similarity
        ## compute similarity of each user to the provided user
        user_sim=self.user_item.loc[user_id,:].dot(self.user_item.T)
        ## sort by similarity
        user_sim=user_sim.sort_values(ascending=False).reset_index()

        # remove the own user's id
        user_sim=user_sim.loc[~(user_sim.user_id==user_id),:]
        # calculate  num_interactions
        user_articles=self.user_item.sum(axis=1).reset_index()
        user_sim=user_sim.merge(user_articles, how='left',on='user_id')
        user_sim.columns=['neighbor_id','similarity','num_interactions']
        # sort the neighbors_df by the similarity and then by number of interactions
        neighbors_df=user_sim.sort_values(['similarity','num_interactions'],ascending=False)
        return neighbors_df # Return the dataframe specified in the doc_string


    def user_advance_recs(self, user_id, top_n=10):
        '''
        INPUT:
        user_id - (int) a user id
        top_n - (int) the number of recommendations you want for the user

        OUTPUT:
        recs - (list) a list of recommendations for the user by article id
        rec_names - (list) a list of recommendations for the user by article title

        Description:
        Loops through the users based on closeness to the input user_id
        For each user - finds articles the user hasn't seen before and provides them as recs
        Does this until m recommendations are found

        Notes:
        * Choose the users that have the most total article interactions
        before choosing those with fewer article interactions.

        * Choose articles with the articles with the most total interactions
        before choosing those with fewer total interactions.

        '''
        if top_n != 10:
            self.top_n = top_n

        # define variable:
        closeness_user_ids=[]
        current_user_reading_ids=[]
        closeness_user_reading_ids=[]
        rec_aids=[]
        rec_aids_names=[]


        # df_neighbors sort the neighbors_df by the similarity and then by number of interactions where \
        # highest of each is higher in the dataframe
        df_neighbors=self.get_top_sorted_users(user_id)
        # According to user_id to find out the most similar user_ids list(closeness_user_ids)
        closeness_user_ids=df_neighbors['neighbor_id'].values.tolist()
        # Get the current user's reading  ids list(current_user_reading_ids)
        current_user_reading_ids,_=self.get_user_articles(user_id)
        # For each user - finds articles the user hasn't seen before and provides them as recs
        for u_id in closeness_user_ids:
            ## Get the reading id lists(closeness_reading_ids) of each user in the closeness_user_reading_ids
            closeness_user_reading_ids,_=self.get_user_articles(u_id)
            ## add into rec_aids if closeness_user_reading_ids not in current_user_reading_ids and m < 10
            np_diff=np.setdiff1d(closeness_user_reading_ids,current_user_reading_ids)
            diff=list(np.setdiff1d(np_diff,rec_aids))
            if len(diff) > 0:
                sorted_uid_aids,sorted_uid_names=self.get_top_sorted_articles(diff)
                rec_aids +=sorted_uid_aids
                rec_aids_names += sorted_uid_names
            if len(rec_aids)>self.top_n:
                break
        recs=rec_aids[:self.top_n]
        rec_names=rec_aids_names[:self.top_n]
        return recs, rec_names


if __name__ == '__main__':
    from data_clean import Data_Clean
    from ucfrecommender import UCFRecommender

    #instantiate data clean
    dc = Data_Clean()
    print(dc.articles_clean.shape, dc.interacts_clean.shape)

    ucfr=UCFRecommender(dc.interacts_clean)
    # Quick spot check just use it to test your functions
    rec_ids  = ucfr.user_recs(20)
    print("The top 10 recommendations for user 20 are the following article ids:")
    print(rec_ids)

    # Quick spot check just use it to test your functions
    rec_ids, rec_names = ucfr.user_advance_recs(20)
    print("The top 10 recommendations for user 20 are the following article ids:")
    print(rec_ids)
    print()
    print("The top 10 recommendations for user 20 are the following article names:")
    print(rec_names)
