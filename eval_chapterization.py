# script to automatically evaluate different sets of parameters in the chapterization process

import os, sys, fasttext, fasttext.util, segeval
sys.path.append('./podcast_chapterize')

# change sys path to allow realtive imports of podcast-chapterize modules
import sys
from podcast_chapterize.chapterize.preprocessor_helper import lemma

def eval():

    transcripts = get_transcripts()

    param_matrix = parameter_matrix()

    # load ft models once globally to prevent need for loading the models multiple times
    model_path = fasttext.util.download_model('en', if_exists='ignore')
    ft_en = fasttext.load_model(model_path)

    model_path = fasttext.util.download_model('de', if_exists='ignore')
    ft_de = fasttext.load_model(model_path)

    eval_score_matrix = [[] for i, x in enumerate(param_matrix)] # transcript scores by parameter set

    for i, transcript in enumerate(transcripts):
        true_chapter_boundaries = get_true_chapter_boundaries(transcript)

        print('gold chapter boundaries: ', true_chapter_boundaries)

        # bulk lemmatize to prevent redundancy 
        transcript['tokens'] = lemmatize(transcript['tokens'], transcript['language'])

        for j, params in enumerate(param_matrix):
            print(f'test transcript {i} with parameters {j}')
            boundaries = run_chapterization(transcript, params, ft_en, ft_de)

            score = eval_segmentation(boundaries, true_chapter_boundaries, len(transcript['tokens']))
            print(f'segmentation similarity: {score}')
            eval_score_matrix[j].append(score)
            


def get_transcripts():
    """fetch transcripts from json files in transcripts/ folder and convert tokens to TranscriptToken objects

    Returns:
        list: transcripts
    """    

    import glob, os, json
    from podcast_chapterize.transcribe.SpeechToTextModules.SpeechToTextModule import TranscriptToken

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


class ChapterizerParameter:
    """class that holds parameter configuration for the chapterizer
    """
    def __init__(
        self,
        window_width,
        with_boundaries,
        tfidf_min_df,
        tfidf_max_df,
        savgol_window_length,
        savgol_polyorder,
        doc_vectorizer
    ):
        self.window_width = window_width
        self.with_boundaries = with_boundaries
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.savgol_window_length = savgol_window_length
        self.savgol_polyorder = savgol_polyorder
        self.doc_vectorizer = doc_vectorizer


def parameter_matrix():
    """create a parameter matrix (list of parameter dicts) for every combination of possible parameters
    
    Returns:
        [list of ChapterizerParameter]: parameter configurations
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
    param_matrix = list(itertools.product(*[
        window_width,
        with_boundaries,
        tfidf_min_df,
        tfidf_max_df,
        savgol_window_length,
        savgol_polyorder,
        doc_vectorizer
    ]))

    param_objects = [ChapterizerParameter(*params) for params in param_matrix]

    return param_objects

def lemmatize(tokens, language):
    chunk_tokens_lemma = lemma([token.token for token in tokens], language)
    for i, token in enumerate(tokens):
        token.token = chunk_tokens_lemma[i]
    return tokens

def run_chapterization(transcript, params, ft_en, ft_de):
    """run chapterization with the given parameters
    
    Args:
        transcript (list of TranscriptToken): transcript to chapterize
        params (dict): parameters as dictionary

    Returns:
        [list of int]: identified boundaries (positon as token count)
    """   

    from podcast_chapterize.chapterize.chapterizer import Chapterizer

    c = Chapterizer(
        window_width=params.window_width,
        tfidf_min_df=params.tfidf_min_df,
        tfidf_max_df=params.tfidf_max_df,
        savgol_window_length=params.savgol_window_length,
        savgol_polyorder=params.savgol_polyorder,
        doc_vectorizer=params.doc_vectorizer
    )

    concat_chapters, boundaries = c.chapterize(
        transcript['tokens'],
        ft_en,
        ft_de,
        boundaries=[],
        language=transcript['language'],
        skip_lemmatization=True
    )

    return boundaries

def eval_segmentation(segmentation, gold_segmentation, doc_length):
    """scrores a segmentation compared to a gold standard segmentation
    
    Args:
        segmentation (list): segmentation to score
        gold_segmentation (list): segmentation to compare against
        doc_length (int): length of the segmented document
    
    Returns:
        float: score of segmentation
    """

    #todo: gives bad output, only addition etids, how to get single score value?

    boundary_string_seg = boundary_string_from_boundary_indices(segmentation, doc_length)
    boundary_string_gold = boundary_string_from_boundary_indices(gold_segmentation, doc_length)

    return segeval.boundary_edit_distance(boundary_string_seg, boundary_string_gold)

def boundary_string_from_boundary_indices(segmentation, doc_length):
    """converts boundary indices to segeval-compatible boundary strings
    
    Args:
        segmentation (list of int): list of segmentation boundary indices
        doc_length (int): length of the segmented document

    Returns:
        tuple: Boundary string
    """

    segmentation.append(doc_length)

    masses = []
    tokens_in_segment = 0
    current_boundary_index = 0
    for i in range(doc_length):
        if i >= segmentation[current_boundary_index]:
            masses.append(tokens_in_segment)
            tokens_in_segment = 0
            current_boundary_index += 1
        tokens_in_segment += 1

    return segeval.boundary_string_from_masses(masses)


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
