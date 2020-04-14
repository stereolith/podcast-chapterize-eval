# script to automatically evaluate different sets of parameters in the chapterization process

import importlib

def eval():
    transcripts = get_transcripts()

    print(get_true_chapter_boundaries(transcripts[0]))

    parameter_matrix()

def get_transcripts():
    """fetch transcripts from json files and convert tokens to TranscriptToken objects
    """    

    import glob, os, json
    TranscriptToken = importlib.import_module('podcast-chapterize.transcribe.SpeechToTextModules.SpeechToTextModule').TranscriptToken

    transcripts = []
    for file in glob.glob("transcripts/*.json"):
        with open(file) as f:
            print('get ', file)
            transcript = json.load(f)
            transcript['tokens'] = [TranscriptToken.from_dict(token) for token in transcript['tokens']]
            transcripts.append(transcript)

    return transcripts

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


def parameter_matrix():
    """create a parameter matrix (list of parameter dicts)
    """

    # values to test
    window_width = [50, 100, 150, 200, 250, 300]
    with_boundaries = [0]
    tfidf_min_df = [0, 5]
    tfidf_max_df = [0.6, 0.7, 0.8, 0.9]
    savgol_window_length = [0]
    savgol_polyorder = [3, 4, 5]
    doc_vectorizer = ['ft_sif_average', 'tfidf', 'ft_average', 'ft_sum']

    import itertools
    param_lists = list(itertools.product(*[
        window_width,
        with_boundaries,
        tfidf_min_df,
        tfidf_max_df,
        savgol_window_length,
        savgol_polyorder,
        doc_vectorizer
    ]))
    keys = []

    param_dicts = [{keys[i]: val for i, val in enumerate(param_vec)} for param_vec in param_lists]
    print(l[0])
    print



def run_chapterization(transcript, params):
    """run chapterization with the given parameters
    
    Args:
        transcript (list of TranscriptToken): transcript to chapterize
        params (dict): parameters as dictionary
    """

    Chapterizer = importlib.import_module('podcast-chapterize.chapterize.chapterizer').Chapterizer
    

eval()




def chapterize_action(args):
    from transcribe.SpeechToTextModules.SpeechToTextModule import TranscriptToken
    from chapterize.chapterizer import Chapterizer
    from chapterize.chapter_namer import chapter_names

    with open(args.transcript, 'r') as f:
        transcript = json.load(f)

    tokens = [TranscriptToken(token['token'], token['time']) for token in transcript['tokens']]
    boundaries = transcript['boundaries']

    chapterizer = Chapterizer(
        window_width=args.window_width,
        max_utterance_delta=args.max_utterance_delta,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_max_df=args.tfidf_max_df,
        savgol_window_length=args.savgol_window_length,
        savgol_polyorder=args.savgol_polyorder,
    )
    concat_chapters, minima = chapterizer.chapterize(tokens, boundaries, language=args.language)
    print([f"{tokens[minimum].time}" for minimum in minima])

    titles = chapter_names(concat_chapters)
    print(titles)

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