import subprocess
import feedparser
import os

episodes_numbers = list(range(0,3))
podcast_urls = [
    'http://minkorrekt.de/feed/m4a/',
    'https://forschergeist.de/feed/m4a/',
    'https://ukw.fm/feed/mp3/',
    'https://raumzeit-podcast.de/feed/m4a/',
    'https://fokus-europa.de/feed/m4a/',
    'https://der-lautsprecher.de/feed/m4a',
    'https://logbuch-netzpolitik.de/feed/m4a',
    'https://freakshow.fm/feed/m4a',
    'https://cre.fm/feed/m4a',
    'http://atp.fm/episodes?format=rss',
    'https://layout.fm/rss',
    'https://www.relay.fm/clockwise/feed',
    'https://www.kuechenstud.io/lagedernation/feed/aac/'
]
podcast_lang = ['de', 'de', 'de', 'de', 'de', 'de', 'de', 'de', 'de', 'en', 'en', 'en', 'de']

def start():
    for i, podcast in enumerate(podcast_urls):
        for no in episodes_numbers:
            url = get_audio_url(podcast, no)['episodeUrl']
            episode_name = os.path.basename(url)
            if check_duplicate(episode_name) == False:
                append_filename(episode_name)
                print(f'transcribe podcast {podcast} ({podcast_lang[i]}) for episode {no}')
                out = subprocess.run([
                    'python3',
                    '../podcast-chapterize/main.py',
                    'transcribe',
                    '-l',
                    podcast_lang[i],
                    '-c',
                    '-e',
                    str(no),
                    podcast,
                    '.'
                    ], stdout=subprocess.PIPE)
                print(out.stdout)

def append_filename(filename):
    with open('done','a+') as f:
        f.write(filename + '\n')

def check_duplicate(episode_name):
    with open('done') as f:
        done = f.readlines()
    print(episode_name)
    for name in done:
        if episode_name in name:
            return True
    return False

def get_audio_url(feed_url, episode=0):
    feed = feedparser.parse(feed_url)
    try:
        last_episode = feed['entries'][episode]

        audio_url = ''
        for link in last_episode['links']:
            if link['rel'] == 'enclosure':
                audio_url = link['href']

        if audio_url == '':
            print('could not find audio url')
            return None

        if audio_url.rfind('?') != -1:
            audio_url = audio_url[:audio_url.rfind('?')]
        print('episode name: ', last_episode['title'])
        print('audio file url: ', audio_url)
        return {
            'episodeUrl': audio_url,
            'episodeTitle': last_episode['title'],
            'author': last_episode['author'] if 'author' in last_episode else ""
        }
    except IndexError:
        print('could not find feed')
        return None

start()
