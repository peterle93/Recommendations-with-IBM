import pandas as pd
import numpy as np
class Data_Clean():
    '''
    This Data_Clean is used to build the dataset, clean the dataset
    '''
    def __init__(self,  interact_pth='data/user-item-interactions.csv', articles_pth='data/articles_community.csv'):
        '''
        Description: initiate Data_Clean class
        Args:
         interact_pth - path to csv with at least the four columns: article_id,title,email
         articles_pth - path to csv with each movie and movie information in each row
        Return:
         N/A
        '''
        self.interacts_clean = None
        self.articles_clean = None
        self.interacts = pd.read_csv(interact_pth)
        self.articles = pd.read_csv(articles_pth)
        del self.interacts['Unnamed: 0']
        del self.articles['Unnamed: 0']
        email_encoded = self.email_mapper()
        del self.interacts['email']
        self.interacts['user_id'] = email_encoded
        self.interacts_clean, self.articles_clean = self.remove_duplicated()
        self.fill_NaN()

    def email_mapper(self):
        '''
        Description: Run this cell to map the user email to a user_id column and remove the email column
        Args:
          N/A
        Return：
         email_encoded: dict list,key: email, value: user_id
        '''
        coded_dict = dict()
        cter = 1
        email_encoded = []
        for val in self.interacts['email']:
            if val not in coded_dict:
                coded_dict[val] = cter
                cter+=1
            email_encoded.append(coded_dict[val])
        return email_encoded

    def remove_duplicated(self):
        '''
        Description:  Remove any rows that have the same article_id in articles dataset
        Args:
          self.articles
        Return：
          interacts_clean: dataframe
          articles_clean: dataframe
        '''
        # Remove any rows that have the same article_id and user_id in interacts dataset- only keep the first
        interacts_clean=self.interacts.drop_duplicates()
        # Remove any rows that have the same article_id in articles dataset- only keep the first
        articles_clean=self.articles.drop_duplicates(subset=['article_id'])
        return interacts_clean, articles_clean

    def fill_NaN(self):
        '''
        Description:  Remove any rows that have the same article_id in articles dataset
        Args:
          self.articles_clean
        Return：
          self.articles_clean
        '''
        if self.articles_clean is not None:
            #Replace NaN with an empty string
            self.articles_clean = self.articles_clean.fillna('')

if __name__ == '__main__':
    import data_clean as dc

    #instantiate data clean
    pre_dc = dc.Data_Clean()
    print(pre_dc.interacts.head())
    print(pre_dc.articles.head())
    print(pre_dc.articles_clean.shape, pre_dc.interacts_clean.shape)
