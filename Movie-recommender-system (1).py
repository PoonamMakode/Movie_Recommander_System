#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import os




# In[45]:


movies= pd.read_csv('tmdb_5000_movies.csv')
credits= pd.read_csv('tmdb_5000_credits.csv')


# In[46]:


movies.head(3)


# In[47]:


movies.shape



# In[48]:


credits.head(3)

# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)


# In[49]:


credits.head(1)


# In[50]:


#shape of movies data
movies.shape


# In[51]:


# merge data 

movies = movies.merge(credits,on='title')
movies.head(1)


# In[52]:


# genres
#  id
#  keywords
# overview
# cast
# crew

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[53]:


movies.head()


# In[54]:


# shape of merge data 
movies.shape


# In[55]:


movies.info()


# In[56]:


movies.duplicated().sum()


# In[57]:


movies.iloc[0].genres


# In[58]:


import ast


# In[59]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[60]:


movies.dropna(inplace=True)


# In[61]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[62]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[63]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[64]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[65]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[66]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[67]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[68]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[69]:


#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)


# In[70]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[71]:


movies.head()


# In[72]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[73]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[74]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()


# In[75]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[76]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[77]:


vector = cv.fit_transform(new['tags']).toarray()


# In[78]:


vector.shape


# In[79]:


from sklearn.metrics.pairwise import cosine_similarity


# In[80]:


similarity = cosine_similarity(vector)


# In[81]:


sorted(similarity[0],reverse=True)


# In[82]:


sorted(list(enumerate(similarity[0])),reverse=True, key=lambda x:x[1])[1:6]


# In[83]:


similarity


# In[84]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[90]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[91]:


recommend('Avatar')


# In[ ]:


import pickle


# In[ ]:


pickle.dump(new,open('movie_list.pkl','wb'))


# In[ ]:


new_df['title'].values


# In[ ]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[ ]:


pickle.dump(similarity,open('movie_similarity.pkl','wb'))


# In[ ]:




