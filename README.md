# pfizer-vaccine-tweet-lda
Use LDA topic modeling on Pfizer vaccine tweets corpus

Basic steps in topic modeling code:

1. Pre-processing
2. Bi-grams/Tri-grams and Stop Word removals
3. Lemmatization and term frequency
4. Determine optimal number of topics using Mallet
5. Train LDA Model
6. Assign dominant topic per document
7. Basic analytics

## Number of Topics - Optimization with Mallet
![alt text](https://github.com/datavizhokie/pfizer-vaccine-tweet-lda/blob/master/Mallet%20Topic%20Coherence.png)

## Intertopic Distance Map
![alt text](https://github.com/datavizhokie/pfizer-vaccine-tweet-lda/blob/master/intertopic%20distance%20map.png)

## Discussion

Although 8 topics was optimal according to the coherence score in the Mallet run, I chose 6 for better consumer digestion. Overall coherence for the trained LDA job with 6 topics is 0.58. From the intertopic distance visualization, we can see that many of the topics overlap. This data was collected via Tweepy with a distinct hashtag (#PfizerBioNTech). Thus, we cannot expect to have very distinct topics. Still, the coherence score is adequate.
