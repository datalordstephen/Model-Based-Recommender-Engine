import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_option('deprecation.showfileUploaderEncoding',False) 
model = pickle.load(open('knn_recom.pkl','rb'))

# creating a pivot_table
df = pd.read_csv('merged.csv')
user_book_table = df.pivot_table(index = ['Book-Title'], columns = ['User-ID'], values = ['Book-Rating']).fillna(0)
books_list = df['Book-Title'].unique()

def main():
    st.markdown("<h1 style='text-align: center; color: Black;background-color:#f32445'>Book Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: Black;'>Drop in The required Inputs and we will do  the rest :)</h3>", unsafe_allow_html=True)
    st.header("What is this Project about?")
    st.text("It is a Web app that would help the user recommend books related to the one you input.")
    st.header("What tools where used to make this?") 
    st.text("The Model was made using a dataset of Books and Ratings. I made use of Sci-Kit learn's NearestNeighbour model in order to make recommendations.")

# function to find the book's index and rating values
    def find_ratings(book):
        if len(book) > 0:
            book = str(book)
            book = book.strip()
            book = user_book_table.loc[book].values.reshape(1, -1)
            return book
        else:
            st.text('Please enter a book that you want recommendations based on')

    def generate_recommendations(distances, indices):
        print(f'Distances: {distances}')
        print(f'Indices: {indices}')
        print(f'Flattened Distances: {distances.flatten()}')
        book = [] 
        distance = []
        for i in range(0, len(distances.flatten())):
            if i != 0:
                book.append(user_book_table.index[indices.flatten()[i]])
                distance.append(distances.flatten()[i])    
        b=pd.Series(book,name='book')
        d=pd.Series(distance,name='distance')
        recommend = pd.concat([b,d], axis=1)
        recommend = recommend.sort_values('distance',ascending=False)
        for i in range(0,recommend.shape[0]):
            print(st.text ('{0}.) {1} : {2}'.format(i+1, recommend["book"].iloc[i], recommend["distance"].iloc[i])))


    Book = st.sidebar.selectbox("What Book have you enjoyed in the past?", books_list)
    number = st.sidebar.slider('How Many suggestions do you want?',1,10)
    
    if st.sidebar.button('Recommend'):
        input_1 = find_ratings(Book).astype('float64')
        input_2 = number + 1
        st.header(f'Recommendations for "{Book}" :\n')
        distances, indices = model.kneighbors(X = input_1, n_neighbors = input_2)
        generate_recommendations(distances, indices)
        st.balloons()
if __name__ =='__main__':
    main() 