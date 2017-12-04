import trumble as tb

tweets = tb.collect_tweets(download = False)

bigstr = ''

for tweet in tweets:
    bigstr = bigstr + tweet
    
trset = set(bigstr)