# pfizer-vaccine-tweet-lda

## Approach

The goal is to use LDA topic modeling on Pfizer vaccine tweets corpus. "In natural language processing, the Latent Dirichlet Allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar."

https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation

## Basic steps in topic modeling code:

1. Pre-processing
2. Bi-grams/Tri-grams and Stop Word removals
3. Lemmatization and term frequency
4. Determine optimal number of topics using Mallet
5. Train LDA Model
6. Assign dominant topic per document
7. Basic analytics

## Number of Topics - Optimization with Mallet
<img src="https://github.com/datavizhokie/pfizer-vaccine-tweet-lda/blob/master/Mallet%20Topic%20Coherence.png" width=50% height=50%>

## Intertopic Distance Map
<img src="https://github.com/datavizhokie/pfizer-vaccine-tweet-lda/blob/master/intertopic%20distance%20map.png" width=75% height=75%>

## Discussion

Although 8 topics was optimal according to the coherence score in the Mallet run, I chose 6 for better consumer digestion. Overall coherence for the trained LDA job with 6 topics is 0.58. From the intertopic distance visualization, we can see that many of the topics overlap. This data was collected via Tweepy with a distinct hashtag (#PfizerBioNTech). Thus, we cannot expect to have very distinct topics... or advanced lexicons ;) 

![Alt Text](https://media.giphy.com/media/26BRNSms6vm0qR8kg/giphy.gif)

Still, the coherence score is adequate, but this corpus is not fantastic for topic modeling.

*Data Source: https://www.kaggle.com/gpreda/pfizer-vaccine-tweets*
