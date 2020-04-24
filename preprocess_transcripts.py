def prepare_transcripts():
    """Main function to preprocess and prepare transcript files for evaluation. Pickles the prepared transcript object.:
        * check language value (ignore transcripts where language value is not set)
        * bulk lemmatize tokens
        * bulk fasttext vectorization
        * calculate true chapter boundaries
    """    
    import pickle

    transcripts = parse_transcripts_json()
    transcripts = bulk_lemmatize(transcripts)
    transcripts = bulk_fasttext(transcripts)

    for transcript in transcripts:
        transcript['trueChapterBoundaries'] = get_true_chapter_boundaries(transcript)

    pickle.dump(transcripts, open('transcripts/transcripts.pickle', 'wb'))


def bulk_fasttext(transcripts):
    """parses transcripts from json files, bulk lemmatizes, vectorizes tokens and adds true chapter boundary list
    """
    model_path = fasttext.util.download_model('en', if_exists='ignore')
    ft_en = fasttext.load_model(model_path)

    model_path = fasttext.util.download_model('de', if_exists='ignore')
    ft_de = fasttext.load_model(model_path)

    for transcript in transcripts:
        for token in transcript['tokens']:
            if transcript['language'] == 'en':
                token.fasttext = ft_en[token.token]
            elif transcript['language'] == 'de':
                token.fasttext = ft_de[token.token]

    return transcripts

def bulk_lemmatize(transcripts):
    """parses transcripts from json files, bulk lemmatizes, vectorizes tokens and adds true chapter boundary list
    """

    for transcript in transcripts:
        transcript['tokens'] = lemmatize_doc(transcript['tokens'], transcript['language'])

    return transcripts


def lemmatize_doc(tokens, language):
    chunk_tokens_lemma = lemma([token.token for token in tokens], language)
    for i, token in enumerate(tokens):
        token.token = chunk_tokens_lemma[i]
    return tokens

def get_true_chapter_boundaries(transcript):
    """extract indices of transcript tokens for true boundaries from chapter json 
    
    Args:
        transcript (dict): transcript to extract boundarie indices from
    
    Returns:
        list: true boundary indices
    """    
    true_chapter_boundaries = []

    for chapter in transcript['chapters']:
        current_token = 0
        while float(chapter['start_time']) > transcript['tokens'][current_token].time:
            current_token += 1
        true_chapter_boundaries.append(current_token)
    
    return true_chapter_boundaries[1:]

# helper functions
def set_lang(match, language):
    import glob, os, json
    for file in glob.glob("transcripts/*.json"):
        if match in file:
            print(f'added lang {language} to {file}')
            with open(file, 'r') as f:
                j = json.load(f)
            j['language'] = language
            with open(file, 'w') as f:
                json.dump(j, f)

def check_lang():
    import glob, os, json
    for file in glob.glob("transcripts/*.json"):
        with open(file, 'r') as f:
            j = json.load(f)
        try:
            l = j['language']
        except KeyError:
            print(f'no language set for file {file}')
