import math

import numpy as np
import pandas as pd

from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


def algorithm(feature):
    """
        Content-Based Algorithm+

        기존의 Content-Based에서 Youflix 시스템을 위해 튜닝이 된 알고리즘 입니다.
        사용자가 평가한 결과를 토대로 더 높은 점수를 받은 영화는 가중치를 높게줘서
        비슷한 영화가 많이 추천 되도록 하고, 반대로 낮은 점수의 영화와 유사한 영화는
        추천되지 않도록 하여 사용자에게 의미있는 결과를 내도록 하였습니다.

        추가적으로 영화감독이나 영화배우들을 기준으로 추천을 받을 수 있도록 구현을 했으나,
        정확도가 높지는 않습니다. 조금 더 연구가 필요할 것으로 예상됩니다.
    """

    # 전처리한 데이터들을 읽어 옵니다.
    # feature 파라미터 값에 따라 불러오는 파일이 다릅니다.
    if feature == 'Director':
        df_keys = pd.read_csv('df_keys_crew.csv')
    elif feature == 'Actor':
        df_keys = pd.read_csv('df_keys_cast.csv')
    elif feature == 'Director/Actor':
        df_keys = pd.read_csv('df_keys_crew_cast.csv')
    else:
        df_keys = pd.read_csv('df_keys.csv')

    # scikit-learn의 TF-IDF Vectorizer 사용하여 키워드를 토큰화하여 행렬을 분석하여 각 단어의 빈도를 분석합니다.
    tfidf_vectorizer = TfidfVectorizer()
    tf_mx = tfidf_vectorizer.fit_transform(df_keys['keywords'])

    # 영화에 대한 유사도 값을 저장할 배열입니다.
    selected_movies = [[0 for i in range(len(tfidf_vectorizer.get_feature_names()))]]

    """
        사용자의 평가에 따라 가중치 값을 더 줍니다.
        임의로 값을 정하여 곱하기를 해주되 너무 큰 값으로 하면 지나치게 값이 커져
        의도하지 않은 결과가 나올 수 있습니다.
        그러므로 math.exp() 메소드를 이용하여 가중치를 주되 너무 큰 값이 되지 않도록 하였습니다.
        
        테스트에 사용되는 영화는 아래와 같습니다.
        1. Star Wars
        2. Star Wars: Episode I - The Phantom Menace
        3. Star Wars: The Force Awakens
    """
    for rating_movie_id in [11, 1893, 140607]:

        movie_frame = df_keys.loc[df_keys['id'] == rating_movie_id]     # 영화 id를 기준으로 전체에서 찾습니다.

        print(movie_frame['title'])

        idx = int(str(movie_frame['Unnamed: 0']).split(' ')[0])         # 영화의 index를 찾습니다.
        movie = tf_mx[idx: idx + 1]                                     # 전체 Dataframe에서 index행을 가져옵니다.

        col = movie.tocoo().col

        # 실제로는 사용자의 평가 데이터를 기반으로 해야 하지만,
        # 테스트를 위해 3개의 영화 모두 5점을 줬다고 가정했습니다.
        for index in range(len(col)):
            selected_movies[0][col[index]] += 1 * math.exp(5)

    cosine_sim = linear_kernel(selected_movies, tf_mx)                  # Cosine Similarity 알고리즘을 사용하여 유사도를 분석합니다.
    indices = pd.Series(df_keys.index, index=df_keys['id'])             # 일치하는 index list를 생성합니다.
    n_rank_list = recommend_movie(df_keys, indices, cosine_sim, -1)     # 유사도 높은 순으로 n개의 추천 영화를 추출합니다.
    return n_rank_list


def recommend_movie(df_keys, indices, cosine_sim, n):
    """
        유사도 분석으로 나온 값을 토대로,
        추천받고자 하는 영화와 가장 유사한 순서대로 정렬을 해주는 메소드입니다.

        # 첫번째 index의 영화의 경우 활성화된 영화와 같기 때문에 제외합니다.
    """

    # 내림차순으로, Cosine Simirality을 정렬합니다.
    scores = pd.Series(cosine_sim[0]).sort_values(ascending=False)

    # n == -1 : 전체.
    if n == -1:
        top_n_idx = list(scores.iloc[1:].index)
    else:
        top_n_idx = list(scores.iloc[1:n].index)

    # 제목을 추출합니다.
    return df_keys['title'].iloc[top_n_idx]


def preprocessing_for_cb():
    """
        데이터 전처리를 위한 method 입니다.
        데이터는 TMDB와 IMDB에서 Youflix 시스템에 필요한 데이터를 추출하고,
        필요한 데이터는 추가하여 사용하였습니다.
        Youflix는 Django Framework를 사용해서 Back-end를 개발하였습니다.
        그래서 아래에 보시면 movies = Movie.objects.all() 과 같은 코드를 보실 수 있는데,
        개발환경에 맞춰서 변경하시면 됩니다.
    """

    # DB에서 모든 movie 정보를 가져옵니다.
    movies = Movie.objects.all()
    # movies 객체를 DataFrame화 합니다.
    movies_frame = pd.DataFrame(movies.values())

    # 불필요한 Column들을 Drop 해줍니다.
    movies_frame = movies_frame.drop('imdb_id', axis=1)
    movies_frame = movies_frame.drop('adult', axis=1)
    movies_frame = movies_frame.drop('collection_id', axis=1)
    movies_frame = movies_frame.drop('budget', axis=1)
    movies_frame = movies_frame.drop('homepage', axis=1)
    movies_frame = movies_frame.drop('popularity', axis=1)
    movies_frame = movies_frame.drop('poster_path', axis=1)
    movies_frame = movies_frame.drop('backdrop_path', axis=1)
    movies_frame = movies_frame.drop('revenue', axis=1)
    movies_frame = movies_frame.drop('runtime', axis=1)
    movies_frame = movies_frame.drop('status', axis=1)
    movies_frame = movies_frame.drop('tagline', axis=1)
    movies_frame = movies_frame.drop('video', axis=1)
    movies_frame = movies_frame.drop('vote_average', axis=1)
    movies_frame = movies_frame.drop('vote_count', axis=1)
    movies_frame = movies_frame.drop('release_date', axis=1)

    # 각 영화에 대한 Keyword와 Genre를 추출합니다.
    keywords = []
    genres = []
    crews_list = []
    casts_list = []

    # DataFrame을 하나씩 참조하며 해당 영화에 맞는 Keyword와 Genre들을 저장합니다.
    # 없을 경우 ''을 삽입 합니다.
    for element in movies_frame.values:

        # id를 통해 movie 객체를 가져와서 필요한 정보를 추출합니다.
        movie = Movie.objects.get(id=element[0])

        # 감독과 배우 정보를 가져옵니다.
        crews = Crew.objects.filter(movie=movie)
        casts = Cast.objects.filter(movie=movie)

        if len(crews) == 0:
            crews_list.append('')
        else:
            temp = []
            for crew in crews:
                if crew.job == 'Director':
                    temp.append(crew.name)
                    break
            if len(temp) == 0:
                crews_list.append('')
            else:
                crews_list.append(temp)

        if len(casts) == 0:
            casts_list.append('')
        else:
            temp = []
            for cast in casts:
                temp.append(cast.name)
                if len(temp) == 3:
                    break
            casts_list.append(temp)

        if len(movie.keywords.all()) is not 0:

            temp = []

            for keyword in movie.keywords.all():
                temp.append(keyword.name)
            keywords.append(temp)
        else:
            keywords.append('')

        if len(movie.genres.all()) is not 0:

            temp = []

            for genre in movie.genres.all():
                temp.append(genre.name)
            genres.append(temp)
        else:
            genres.append('')

    # 데이터 전처리 및 생성
    movies_frame['crews'] = ''
    movies_frame['casts'] = ''
    # 1. overview가 없는 영화에 대해서 ''로 모두 할당 해줍니다.
    movies_frame['overview'] = movies_frame['overview'].fillna('')
    # 2. Genre 데이터를 삽입합니다.
    movies_frame = movies_frame.assign(genres=genres)
    # 3. Keyword 데이터를 삽입합니다.
    movies_frame = movies_frame.assign(keywords=keywords)
    # 4. Crew 데이터를 삽입합니다.
    movies_frame = movies_frame.assign(crews=crews_list)
    # 5. Cast 데이터를 삽입합니다.
    movies_frame = movies_frame.assign(casts=casts_list)

    # genres, keywords 데이터에 대해서 공백(' ')을 없애줍니다.
    movies_frame['genres'] = movies_frame['genres'].apply(preprocessing_genres)
    movies_frame['keywords'] = movies_frame['keywords'].apply(preprocessing_keyword)

    # Rake(Rapid Automatic Keyword Extraction)을 이용해서 줄거리에 대한 keyword을 추출합니다.
    movies_frame['overview'] = movies_frame['overview'].apply(preprocessing_overview)

    # 앞에서 만든 데이터를 통하여 새로운 DataFrame을 생성합니다.
    df_keys = pd.DataFrame()
    df_keys['title'] = movies_frame['title']
    df_keys['keywords'] = ''
    df_keys['id'] = movies_frame['id']

    # 만들어진 단어들을 하나의 단어 모음으로 만듭니다.
    df_keys['keywords'] = movies_frame.apply(bag_words, axis=1)

    # 지금까지의 결과를 .csv 파일로 저장합니다.
    df_keys.to_csv('df_keys_crew_cast.csv', mode='w')

    # ============================== WARNING ==================================
    # 여기서 부터 아래의 코드를 실행하고 싶을 경우,
    # df_keys_*.csv 파일들 중 하나를 Pandas의 DataFrame으로 가져와야 합니다.
    # =========================================================================

    # scikit-learn의 CountVectorizer를 사용하여 키워드를 토큰화하여 행렬을 조사하여 각 단어의 빈도를 분석합니다.
    tfidf_vectorizer = TfidfVectorizer()
    tf_mx = tfidf_vectorizer.fit_transform(df_keys['keywords'])

    selected_movies = df_keys.loc[df_keys['id'] == 862]         # 테스트를 위해 영화 id가 862인 'Toy Story'를 사용했습니다.
    idx = str(selected_movies['Unnamed: 0']).split(' ')         # 불필요한 문자열 값을 삭제합니다.
    idx = int(idx[0])

    # Cosine Similarity 알고리즘을 사용하여 유사도를 분석합니다.
    cosine_sim = cosine_similarity(tf_mx[idx: idx + 1], tf_mx)

    # 일치하는 index list를 생성합니다.
    indices = pd.Series(df_keys.index, index=df_keys['id'])

    # id 597에 대한 상위 10개의 추천 영화를 추출합니다.
    # print(test_recommend_movie(df_keys, 862, indices, 10, cosine_sim))


def test_recommend_movie(df_keys, movie_id, indices, n, cosine_sim):
    movies = []
    # 일치하는 영화 제목 색인을 검색합니다.
    if movie_id not in indices.index:
        print("Movie not in database.")
        return
    else:
        idx = indices[movie_id]
    # 내림차순으로 영화의 코사인 유사성 점수를 정렬합니다.
    scores = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    # 가장 유사한 영화 n개를 추출합니다.
    # index 0은 입력 된 영화와 동일하므로 1부터 사용하시면 됩니다.
    top_n_idx = list(scores.iloc[1:n].index)
    return df_keys['title'].iloc[top_n_idx]


def preprocessing_keyword(data):

    ret = []
    for keyword in data:
        ret.append(keyword.replace(' ', ''))
    return ret


def preprocessing_genres(data):

    ret = []
    for genre in data:
        ret.append(genre.replace(' ', ''))
    return ret


def preprocessing_director(data):

    if data is np.NaN:
        return np.NaN
    ret = data.replace(' ', '')
    return ret


def preprocessing_overview(data):

    plot = data
    rake = Rake()
    rake.extract_keywords_from_text(plot)
    scores = rake.get_word_degrees()
    return(list(scores.keys()))


def bag_words(x):
    return (' '.join(x['genres']) + ' ' + ' '.join(x['keywords']) + ' ' + ' '.join(x['title']) + ' ' + ' '.join(x['overview']) + ' ' + ' '.join(x['crews']) + ' ' + ' '.join(x['casts']))


if __name__ == '__main__':
    print(algorithm('none'))