#!/usr/bin/python

import zipfile
import urllib.request
import html
import json
import re
import numpy as np

URL_F = 'https://github.com/bpb27/trump_tweet_data_archive/raw/master/condensed_{}.json.zip'
TWEET_YEARS = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']

regex = re.compile(r'[\r\n\t]')
hypertext = re.compile(r'http')

def fix_tweets(fnames):
    tweets = []
    for fname in fnames:
        with open(fname) as fp:
            for msg in json.load(fp):
                if msg['is_retweet'] or hypertext.search(msg['text']):
                    continue
                text = regex.sub('', html.unescape(msg['text']))
                tweets.append([text[x] if x < len(text) else '\0' for x in range(140)])
    return np.array(tweets)

def collect_tweets(download = True, years = TWEET_YEARS, dl_years = TWEET_YEARS):
    if download:
        for year in dl_years:
            request = urllib.request.urlopen(URL_F.format(year))
            with open(f'condensed_{year}.json.zip', 'wb') as fp:
                fp.write(request.read())
            with zipfile.ZipFile(f'condensed_{year}.json.zip') as jzip:
                jzip.extractall()
    return fix_tweets([f'condensed_{year}.json' for year in years])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Download Trump Tweets')
    parser.add_argument('years', metavar = 'YYYY', type = str, nargs = '*', default = TWEET_YEARS,
                        help = 'Years to collect, valid from 2009 (default: all years)')
    parser.add_argument('--nodl', dest = 'use_download', action = 'store_const', const = False, default = True,
                        help = 'Set this to disable downloading (default: download new tweets from archive)')
    args = parser.parse_args()

    tweets = collect_tweets(download = args.use_download, years = args.years, dl_years = args.years)
    for tweet in tweets:
        print(tweet)
