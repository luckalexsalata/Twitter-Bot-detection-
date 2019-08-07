from django.shortcuts import render, redirect
from django.http import HttpResponse
from bot_detection.forms import nameForm
import pandas as pd
import numpy as np
from  bot_detection.utils import *

import seaborn as sns
import warnings, time
from sklearn import metrics
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import tweepy

import matplotlib.pyplot as plt
import matplotlib as mpl
from tweepy import OAuthHandler
from datetime import date, datetime




def bot(request):
    #return HttpResponse('Bot_detection')
    form = nameForm()
    return render(request, 'web.html',{'form': form} )

def get_res(request):
    if request.method == 'POST':
        form = nameForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['screen_name']
            algoritm = form.cleaned_data['field']
            #context = {'form': form, 'text': text}
            is_bot, pie_chart_s = machine_learning(text, algoritm)
            #tweets = api.user_timeline(screen_name=text, count=200, include_rts=True, tweet_mode='extended')
            #pie_chart_s.plot.pie(fontsize=11, autopct='%.2f', figsize=(7, 7))
            #img_1 = pie_chart_s.Show()
            #response = HttpResponse(content_type="image/jpeg")
            #pie_chart_s.print_png(response)
            context = {'form': form, 'text': is_bot}
            return render(request, 'web.html', context)
    else:
        form = nameForm()
        context = {'form': form}
        return render(request, 'web.html', context)

def machine_learning(screen_name,algoritm):
    ckey = "hMFgSTgDJ4aG33lCW83RLH6Pp"
    csecret = "GbebitCzPHcpGXE5LpevOLXlAR75kn3U4mhfobgoBVRkuVDJbS"
    atoken = "932671235628109830-KS9cN9NFFmZ60NvRGyeMkzHry2FovdM"
    asecret = "a9b4esDiBl1lufyxaFb7uQv19I7vOND9g3k59wBrykGRU"
    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)
    api = tweepy.API(auth)
    st = api.user_timeline(screen_name = screen_name, count=1, include_rts=True)
    tweets = api.user_timeline(screen_name = screen_name, count=200, include_rts=True, tweet_mode='extended')

    def json_serial(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError("Type %s not serializable" % type(obj))

    # print (user)
    for status in st:
        # print (status)#.user.created_at,status.user.name, status.user.screen_name,status.user.description,status.user.verified,status.user.location,status.user.followers_count,status.user.friends_count,status.user.listed_count,status.user.favourites_count,status.user.statuses_count)
        user_id = status.user.id
        created_at = status.user.created_at
        name = status.user.name
        screen_name = status.user.screen_name
        description = status.user.description
        verified = status.user.verified
        location = status.user.location
        followers_count = status.user.followers_count
        friends_count = status.user.friends_count
        listed_count = status.user.listed_count
        favourites_count = status.user.favourites_count
        statuses_count = status.user.statuses_count
        default_profile = status.user.default_profile
        default_profile_image = status.user.default_profile_image
        has_extended_profile = status.user.has_extended_profile

    data = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=['Tweets'])
    #################################################################
    a = created_at
    b = tweets[0].created_at
    delta = b - a
    tweets_per_day = (statuses_count / delta.days)
    ###################################################################
    data['Source'] = np.array([tweet.source for tweet in tweets])
    sources = []
    for source in data['Source']:
        if source not in sources:
            sources.append(source)
    sources_count = len(sources)
    for source in sources:
        sources_count = sources_count + 1
    percent_s = np.zeros(len(sources))
    pie_chart_s = pd.Series(percent_s, index=sources, name='')


    ##########################################################
    languages = []
    data['lang'] = np.array([tweet.lang for tweet in tweets])
    for language in data['lang']:
        if language not in languages:
            languages.append(language)
    languages_count = len(languages)
    ###########################################################
    retweeted = []
    is_there_false = 0
    is_there_true = 0
    for status in tweets:
        try:
            status.retweeted_status
        except AttributeError:
            retweeted.append(False)
            is_there_false = 1
        else:
            retweeted.append(True)
            is_there_true = 1
    how_many_elements = is_there_true + is_there_false
    print(retweeted)
    data['retweeted'] = np.array(retweeted)
    retweeteds = []
    for retweeted in data['retweeted']:
        if retweeted not in retweeteds:
            retweeteds.append(retweeted)
    percent = np.zeros(len(retweeteds))
    for retweeted in data['retweeted']:
        for index in range(len(retweeteds)):
            if retweeted == retweeteds[index]:
                percent[index] += 1
                pass

    pie_chart = pd.Series(percent, index=retweeteds, name='retweeted')
    l = len(tweets)
    if how_many_elements == 2:
        if data['retweeted'][0] == True:
            retweeteds_percent = percent[0] / l
        else:
            retweeteds_percent = percent[1] / l
    else:
        if data['retweeted'][0] == True:
            retweeteds_percent = percent[0] / l
        else:
            retweeteds_percent = 1 - percent[0] / l
    ####################################################################
    geo_ = []
    data['geo'] = np.array([tweet.geo for tweet in tweets])
    for geo in data['geo']:
        if geo not in geo_:
            geo_.append(geo)
    geo_count = len(geo_)

    ######################################################################
    coordinates_ = []
    data['coordinates'] = np.array([tweet.coordinates for tweet in tweets])
    for coordinates in data['coordinates']:
        if coordinates not in coordinates_:
            coordinates_.append(coordinates)
    coordinates_count = len(coordinates_)
    ######################################################################
    from textblob import TextBlob
    import re

    def clean_tweet(tweet):
        '''
        Utility function to clean the text in a tweet by removing
        links and special characters using regex.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analize_sentiment(tweet):
        '''
        Utility function to classify the polarity of a tweet
        using textblob.
        '''
        analysis = TextBlob(clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    data['SA'] = np.array([analize_sentiment(tweet) for tweet in data['Tweets']])
    pos_tweets = len([tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]) / len(
        data['Tweets'])
    neu_tweets = len([tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]) / len(
        data['Tweets'])
    neg_tweets = len([tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]) / len(
        data['Tweets'])

    ################################################################################
    user =  pd.DataFrame({
        'user_id': [user_id],
        'created_at': [created_at],
        'name': [name],
        'screen_name': [screen_name],
        'description': [description],
        'verified': [verified],
        'location': [location],
        'followers_count': [followers_count],
        'friends_count': [friends_count],
        'listed_count': [listed_count],
        'favourites_count': [favourites_count],
        'statuses_count': [statuses_count],
        'default_profile': [default_profile],
        'default_profile_image': [default_profile_image],
        'has_extended_profile': [has_extended_profile],
        'tweets_per_day': [tweets_per_day],
        'sources_count': [sources_count],
        'languages_count': [languages_count],
        'retweeteds_percent': [retweeteds_percent],
        'geo_count': [geo_count],
        'coordinates_count': [coordinates_count],
        'pos_tweets': [pos_tweets],
        'neu_tweets': [neu_tweets],
        'neg_tweets': [neg_tweets]
    })
    if algoritm == '1':
        alg = Knn_alg(user)
    if algoritm == '2':
        alg = Logistic_Regression(user)
    if algoritm == '3':
        alg = TREE(user)
    if algoritm == '4':
        alg = Random_Forest(user)
    if algoritm == '5':
        alg = Gradient_Boosting(user)
    if algoritm == '6':
        alg = neural_network(user)
    if algoritm == '7':
        alg = my_algoritm(user)





    #alg = Knn_alg(user)
    #alg = Logistic_Regression(user)
    #alg = TREE(user)
    #alg = Random_Forest(user)
    # alg = Gradient_Boosting(user)

    #alg = neural_network(user)
    #alg = my_algoritm(user)
    return alg, pie_chart_s

def my_algoritm(training_data):
    bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                       r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                       r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                       r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
    training_data['name_verification'] = training_data.name.str.contains(bag_of_words_bot, case=False, na=False)
    training_data['screen_name_verification'] = training_data.screen_name.str.contains(bag_of_words_bot, case=False,
                                                                                       na=False)
    training_data['description_verification'] = training_data.description.str.contains(bag_of_words_bot, case=False,
                                                                                       na=False)
    i = 0
    h = 0
    while i < 1:
        if training_data.tweets_per_day[i] > 7.7:

            if training_data.retweeteds_percent[i] > 0.85:
                a_0 = 0.25
            else:
                if training_data.retweeteds_percent[i] == 0:
                    a_0 = 0.4
                    if training_data.tweets_per_day[i] > 20:
                        a_0 = 0.5
                else:
                    a_0 = training_data.tweets_per_day[i] / 100

                if training_data.tweets_per_day[i] > 25:
                    a_0 = 0.5
        else:
            if training_data.retweeteds_percent[i] > 0.9:
                a_0 = 0.25
            else:
                if training_data.retweeteds_percent[i] == 0:
                    a_0 = 0.2
                else:
                    a_0 = 0

        if training_data.verified[i] == True:
            a_1 = -2
        else:
            a_1 = 0
        # if training_data.default_profile[i] == True:
        #    a_2 = 0.3
        ####################################################################
        if training_data.languages_count[i] > 4:
            a_3 = 0.05
            if training_data.languages_count[i] > 5:
                a_3 = 0.1
                if training_data.languages_count[i] > 6:
                    a_3 = 0.15
                    if training_data.languages_count[i] > 7:
                        a_3 = 0.25
                        if training_data.languages_count[i] > 9:
                            a_3 = 0.4
        else:
            a_3 = 0

        if training_data.sources_count[i] >= 3:
            a_4 = 0
            if training_data.sources_count[i] >= 4:
                a_4 = - 0.1
                if training_data.sources_count[i] >= 5:
                    a_4 = -0.3


        else:
            if training_data.sources_count[i] == 2:
                a_4 = 0.1
            else:
                a_4 = 0.2

        ###########################################################
        if training_data.friends_count[i] == 0:
            if training_data.followers_count[i] > 100:
                a_5 = 0.4
                if training_data.followers_count[i] > 10000:
                    a_5 = 0.5
            else:
                if training_data.followers_count[i] < 10:
                    a_5 = 0
        else:
            if training_data.friends_count[i] < training_data.followers_count[i] / 100:
                a_5 = 0.25
            else:
                a_5 = 0
        #############################################333333333333333333333
        if training_data.name_verification[i] == True:
            a_6 = 0.2
            if training_data.screen_name_verification[i] == True:
                a_6 = 0.4
                if training_data.description_verification[i] == True:
                    a_6 = 0.5
            else:
                if training_data.description_verification[i] == True:
                    a_6 = 0.4
        else:
            if training_data.screen_name_verification[i] == True:
                a_6 = 0.2
                if training_data.description_verification[i] == True:
                    a_6 = 0.4
            else:
                if training_data.description_verification[i] == True:
                    a_6 = 0.2
                else:
                    a_6 = 0
        ###################################################################3333333
        if training_data.pos_tweets[i] > 0.1:
            a_7 = 0
            if training_data.pos_tweets[i] > 0.8:
                a_7 = 0.1
        else:
            a_7 = 0.1
        if training_data.neg_tweets[i] == 1:
            if training_data.statuses_count[i] != 1:
                a_8 = 0
        ##############################
        c = a_0 + a_1 + a_3 + a_4 + a_5 + a_6 + a_7
        # print(c)
        if c >= 0.5:
            y = 'it is bot'
        else:
            y = 'it is real user'
        i = i + 1
    # print (y)

    return y
def bed_words(training_data):
    bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                       r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                       r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                       r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
    training_data['name_verification'] = training_data.name.str.contains(bag_of_words_bot, case=False, na=False)
    training_data['screen_name_verification'] = training_data.screen_name.str.contains(bag_of_words_bot, case=False,
                                                                                       na=False)
    training_data['description_verification'] = training_data.description.str.contains(bag_of_words_bot, case=False,
                                                                                       na=False)
    return training_data
def get_data():

    training_data = pd.read_json('C:/test.json', encoding="utf8", orient='columns')
    bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                       r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                       r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                       r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
    training_data['name_verification'] = training_data.name.str.contains(bag_of_words_bot, case=False, na=False)
    training_data['screen_name_verification'] = training_data.screen_name.str.contains(bag_of_words_bot, case=False,
                                                                                       na=False)
    training_data['description_verification'] = training_data.description.str.contains(bag_of_words_bot, case=False,
                                                                                       na=False)
    features = ['verified', 'followers_count',  'friends_count', 'statuses_count', 'coordinates_count', 'geo_count',
                'tweets_per_day', 'sources_count', 'languages_count', 'retweeteds_percent', 'pos_tweets', 'neu_tweets',
                'neg_tweets']
    #'name_verification', 'description_verification','screen_name_verification',
    X = training_data[features]
    # print("Форма массива data: {}".format(X.shape))
    from sklearn.metrics import accuracy_score, roc_curve, auc
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, training_data.bot, random_state=0)
    # вычисляем среднее для каждого признака обучающего набора
    #mean_on_train = X_train.mean(axis=0)
    # вычисляем стандартное отклонение для каждого признака обучающего набора
    #std_on_train = X_train.std(axis=0)
    # вычитаем среднее и затем умножаем на обратную величину стандартного отклонения
    # mean=0 и std=1
    #X_train_scaled = (X_train - mean_on_train) / std_on_train
    # используем ТО ЖЕ САМОЕ преобразование (используем среднее и стандартное отклонение
    # обучающего набора) для тестового набора
    #X_test_scaled = (X_test - mean_on_train) / std_on_train
    return features, X_train, y_train

def Knn_alg(user):
    features, X_train_scaled, y_train  =  get_data()
    user = bed_words(user)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    X_new = user[features]
    prediction = knn.predict(X_new)
    if prediction == 0:
        text = 'kNN: This is real user'
    if prediction == 1:
        text = 'kNN: This is bot'
    return text


def Logistic_Regression(user):
    features, X_train_scaled, y_train = get_data()
    user = bed_words(user)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    logreg = LogisticRegression().fit(X_train_scaled, y_train)
    X_new = user[features]
    prediction = logreg.predict(X_new)
    if prediction == 0:
        text = 'Logistic Regression: This is real user'
    if prediction == 1:
        text = 'Logistic Regression: This is Bot'
    return text

def TREE(user):
    features, X_train_scaled, y_train = get_data()
    user = bed_words(user)
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(X_train_scaled, y_train)
    X_new = user[features]
    prediction = tree.predict(X_new)
    if prediction == 0:
        text = 'Decision Tree: This is real user'
    if prediction == 1:
        text = 'Decision Tree: This is Bot'
    return text



def Random_Forest(user):
    features, X_train_scaled, y_train = get_data()
    user = bed_words(user)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(criterion='entropy', min_samples_leaf=100, min_samples_split=29)
    rf.fit(X_train_scaled, y_train)
    X_new = user[features]
    prediction = rf.predict(X_new)
    if prediction == 0:
        text ='Random Forest: This is real user'
    if prediction == 1:
        text = 'Random Forest: This is bot'
    return text


def Gradient_Boosting(user):
    features, X_train_scaled, y_train = get_data()
    user = bed_words(user)
    from sklearn.ensemble import GradientBoostingClassifier
    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train_scaled, y_train)
    X_new = user[features]
    prediction = gbrt.predict(X_new)
    if prediction == 0:
        text = 'Gradient Boosting: This is real user'
    if prediction == 1:
        text = 'Gradient Boosting: This is bot'
    return text

def neural_network(user):
    features, X_train_scaled, y_train = get_data()
    user = bed_words(user)
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
    mlp.fit(X_train_scaled, y_train)
    X_new = user[features]
    prediction = mlp.predict(X_new)
    if prediction == 0:
        text = 'MLP: This is real user'
    if prediction == 1:
        text = 'MLP: This is bot'
    return text