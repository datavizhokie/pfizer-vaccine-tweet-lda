import nltk
import re
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore",category=DeprecationWarning)
#     import imp

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import sys


### CONFIGS
mallet_path     = '/Users/matt.wheeler/mallet/mallet-2.0.8/bin/mallet'
grams           = 'bigrams'
###


def read_data(source_file):

    df = pd.read_csv(source_file)

    print(f"There are {len(df)} tweets in the imported dataset.")
    print("raw data:")
    print(df.head())

    return df


def preprocess(df, desc_field):

    """
    Basic NLP preprocessing (remove non-alphabet characters, remove short words, make all lowercase)

    Parameters:
    ----------
    df : source dataframe
    desc_field: field from source data that contains the descriptions

    Returns:
    -------
    desc_list: list of list of processed words
    """

    # removing everything except alphabets`
    df['clean_desc'] = df[desc_field].str.replace("[^a-zA-Z#]", " ")

    # removing short words
    df['clean_desc'] = df['clean_desc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

    # make all text lowercase
    df['clean_desc'] = df['clean_desc'].apply(lambda x: x.lower())

    # Description to a list
    desc_list = df.clean_desc.values.tolist()

    return desc_list


def grams_and_stop_removals(grams, word_list):

    """
    1. Remove stopwords 
    2. Use gensim.models.Phrases to create Bigrams and Trigrams

    Parameters:
    ----------
    grams : bigrams or trigrams
    word_list: list of list of words from descriptions

    Returns:
    -------
    grams_selected : bigrams or trigrams, based on specification
    data_words_nostops: list of list of words, minus stopwords
    """

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(word_list, min_count=5, threshold=10) # min_count = min frequency for word/gram; higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[word_list], threshold=10)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    stop_words.extend(['tweet','pfizerbiontech','pfizervaccine','covidvaccine','vaccine','https'])
    data_words_nostops = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in word_list]

    # make grams
    bigrams = [bigram_mod[doc] for doc in data_words_nostops]
    trigrams = [trigram_mod[bigram_mod[doc]] for doc in data_words_nostops]

    if grams == 'bigrams':
        return bigrams, data_words_nostops
    else:
        return trigrams, data_words_nostops


def lemmatize_and_term_freq(grams_selected):

    """
    1. Lemmatize bigrams or trigrams (removes tenses, etc)
    2. Obtain Document Term Frequency matrix for modeling

    Parameters:
    ----------
    grams_selected : bigrams or trigrams

    Returns:
    -------
    data_lemmatized : List of lists of lemmatized sentences
    corpus : Term Document Frequency matrix
    id2word : Dictionary of Word and corresponding ID
    """

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    #TODO: consider other articles to keep?
    """https://spacy.io/api/annotation"""
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    data_lemmatized = []
    for sent in grams_selected:
        doc = nlp(" ".join(sent)) 
        data_lemmatized.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    print(f"First 3 Record examples after Lemmatization with *{grams}*: ")
    print(data_lemmatized[:3])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Term Document Frequency (corpus for downstream)
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]

    # Human readable format of corpus (term-frequency)
    print("Example of (Actual Word, Word Freq)...")
    print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

    return data_lemmatized, corpus, id2word


def compute_mallet_coherence_values(mallet_path, corpus, dictionary, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    mallet_path : mallet package location
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    start: Min num of topics
    limit : Max num of topics
    step : step for increments

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path=mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    print("model list:")
    print(model_list)

    return model_list, coherence_values


def viz_mallet_results(start, limit, step, coherence_values, model_list):

    """
    Viz mallet results

    Parameters:
    ----------
    start : min topics from mallet 
    stop : max topics from mallet
    step : num topics step between increments
    coherence_values : coherence scores from all sequential topic models

    Returns:
    -------
    """

    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
        
    # Select the model and print the topics
    # optimal_model = model_list[3]
    # model_topics = optimal_model.show_topics(formatted=False)
    # pprint(optimal_model.print_topics(num_words=10))


def train_topic_model(corpus, id2word, texts, num_topics, chunksize, iterations, passes, alpha):

    """
    Train single LDA model and save pyLDAvis HTML visualization file

    Parameters:
    ----------
    corpus : Gensim corpus
    id2word : Dictionary of Word and corresponding ID
    texts: derp
    num_topics : number of topics
    chunksize: number of documents to be used in each training chunk
    passes : how many iterations for training
    alpha : affects sparsity of topics (auto = 1.0/num_topics)

    Returns:
    -------
    lda_model : model object
    """

    print(f"LDA Model training with {num_topics} topics...")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics, 
                                            random_state=123,
                                            update_every=1,
                                            chunksize=chunksize,
                                            iterations=iterations,
                                            passes=passes,
                                            alpha=alpha,
                                            per_word_topics=True)

    # Print the Keyword in the topics
    print("Resulting LDA Topics:")
    pprint(lda_model.print_topics())

    # Compute Perplexity
    print('\nModel Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nModel Coherence Score (desire above .45!): ', coherence_lda)

    # Visualize the topics with pyLDAvis
    viz = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(viz, 'LDA_Visualization.html')

    return lda_model


def format_topics_sentences(ldamodel, corpus, texts):

    """
    Obtain dominant topic for each document

    Parameters:
    ----------
    ldamodel : model object
    corpus : Gensim corpus
    texts : raw list of descriptions for assessing
   
    Returns:
    -------
    df_topic_sents_keywords : dataframe with assigned topic, along with topic keywords
    """

    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)


def topic_analytics_and_export(df, df_topic):

    # re-format df
    df_dominant_topic = df_topic.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    print("Dominant topic per document:")
    print(df_dominant_topic.head(10))

    # Number of Documents for Each Topic
    topic_counts = df_topic['Dominant_Topic'].value_counts().sort_index()

    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)

    # Topic Number and Keywords
    topic_num_keywords = df_topic.drop_duplicates(['Dominant_Topic', 'Topic_Keywords'])[['Dominant_Topic','Topic_Keywords']].sort_values(by='Dominant_Topic').reset_index(drop=True)

    # Concatenate Column wise
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

    # Change Column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

    print("Topic Distribution Across Documents")
    print(df_dominant_topics)

    # join raw data to dominant topic DF and write to CSV
    df_final = pd.concat([df, df_dominant_topic], axis=1)
    df_final.to_csv('Tweet Data with Topics.csv', index=False)



def main():

    df = read_data('vaccination_tweets.csv')

    # preprocess descriptions
    desc_list = preprocess(df, 'text')

    # Tokenize preprocessed descriptions
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(desc_list))
    print("After tokenization, example:")
    print(data_words[:1])

    # Stop Removals and Grams
    grams_selected, data_words_nostops = grams_and_stop_removals(grams=grams, word_list=data_words)

    print(f"First 3 Record examples after gramming with *{grams}*: ")
    print(grams_selected[:3])

    # Lemmatize and Term Freqs
    data_lemmatized, corpus, id2word = lemmatize_and_term_freq(grams_selected)


    ###### Explore optimal number of topics ######
    # limit=12 # max topics
    # start=4 # min topics
    # step=1
    # model_list, coherence_values = compute_mallet_coherence_values(mallet_path=mallet_path, corpus=corpus, dictionary=id2word, texts=data_lemmatized, limit=limit, start=start, step=step)
    # viz_mallet_results(start=start, limit=limit, step=step, coherence_values=coherence_values, model_list=model_list)
    ##############################################

    # Train single LDA model
    # alpha = auto, symmetric, asymmetric
    lda_model = train_topic_model(corpus=corpus, id2word=id2word, texts=data_lemmatized, num_topics=6, chunksize=40, iterations=20, passes=100, alpha='auto')
    # Apply model to Description list to get optimal topic per document
    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=desc_list)
    # Analytics and export
    topic_analytics_and_export(df=df, df_topic=df_topic_sents_keywords)


if __name__== "__main__" :
    main()