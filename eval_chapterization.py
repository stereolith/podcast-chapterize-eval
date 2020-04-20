# script to automatically evaluate different sets of parameters in the chapterization process

import os, sys, json, fasttext, fasttext.util, segeval
sys.path.append('./podcast_chapterize')

# context to silence stdout
import contextlib
import sys

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

# load ft models once globally to prevent need for loading the models multiple times
model_path = fasttext.util.download_model('en', if_exists='ignore')
ft_en = fasttext.load_model(model_path)

model_path = fasttext.util.download_model('de', if_exists='ignore')
#ft_de = fasttext.load_model(model_path)
ft_de = None

# change sys path to allow realtive imports of podcast-chapterize modules
import sys
from podcast_chapterize.chapterize.preprocessor_helper import lemma


def eval():
    from multiprocessing import Pool
    from functools import partial

    transcripts = get_transcripts()

    param_matrix = parameter_matrix()


    eval_score_matrix = [[] for i, x in enumerate(param_matrix)] # transcript scores by parameter set

    for i, transcript in enumerate(transcripts):
        
        if transcript['language'] == 'en':
            true_chapter_boundaries = get_true_chapter_boundaries(transcript)
            print('gold chapter boundaries: ', true_chapter_boundaries)

            # bulk lemmatize to prevent redundancy 
            transcript['tokens'] = lemmatize(transcript['tokens'], transcript['language'])

            # use multiprocesses to distribute over multiple cores
            pool = Pool()
            get_score_for_current_transcript = partial(get_score, transcript=transcript, true_chapter_boundaries=true_chapter_boundaries, i_transcript=i, len_transcripts=len(transcripts), len_params=len(param_matrix))
            scores = pool.map(get_score_for_current_transcript, enumerate(param_matrix))
            print(scores)

            for j, score in enumerate(scores):
                eval_score_matrix[j].append(score)

    with open('results.json', 'w') as f:
        j = {
            'parameter_matrix': [param.to_dict() for param in param_matrix],
            'results': eval_score_matrix
        }
        json.dump(j, f)
    

def get_score(params_tuple, transcript, true_chapter_boundaries, i_transcript, len_transcripts, len_params):
    params = params_tuple[1]
    i_params = params_tuple[0]
    print(f'test transcript {i_transcript}/{len_transcripts} with parameters {i_params}/{len_params}')
    with nostdout():
        boundaries = run_chapterization(transcript, params, ft_en, ft_de)
        score = eval_segmentation(boundaries, true_chapter_boundaries, len(transcript['tokens']))
    return score


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
            transcript = json.load(f)
        try:
            l = transcript['language']
            c = transcript['chapters']
            transcript['tokens'] = [TranscriptToken.from_dict(token) for token in transcript['tokens']]
            transcripts.append(transcript)
        except:
            print(f'skipping transcript {file} because it is missing language and/or chapter values')

    print(f'fetched {len(transcripts)} transcripts')

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
    
    def to_dict(self):
        return {
            'window_width': self.window_width,
            'with_boundaries': self.with_boundaries,
            'tfidf_min_df': self.tfidf_min_df,
            'tfidf_max_df': self.tfidf_max_df,
            'savgol_window_length': self.savgol_window_length,
            'savgol_polyorder': self.savgol_polyorder,
            'doc_vectorizer': self.doc_vectorizer
        }


def parameter_matrix():
    """create a parameter matrix (list of parameter dicts) for every combination of possible parameters
    
    Returns:
        [list of ChapterizerParameter]: parameter configurations
    """

    # values to test
    window_width = [50, 100, 200, 300]
    with_boundaries = [0]
    tfidf_min_df = [0, 5]
    tfidf_max_df = [0.7, 0.8, 0.9]
    savgol_window_length = [0]
    savgol_polyorder = [4, 5]
    doc_vectorizer = ['tfidf', 'ft_sif_average', 'ft_average', 'ft_sum']

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

    boundary_string_seg = boundary_string_from_boundary_indices(segmentation, doc_length)
    boundary_string_gold = boundary_string_from_boundary_indices(gold_segmentation, doc_length)

    bed = segeval.boundary_edit_distance(boundary_string_seg, boundary_string_gold, n_t=100)

    b = segeval.boundary_similarity(boundary_string_seg, boundary_string_gold, boundary_format=segeval.BoundaryFormat.sets)

    return b

def boundary_string_from_boundary_indices(segmentation, doc_length):
    """converts boundary indices to segeval-compatible boundary strings
    
    Args:
        segmentation (list of int): list of segmentation boundary indices
        doc_length (int): length of the segmented document

    Returns:
        tuple: Boundary string
    """

    i = 1
    tokens_in_segment = 0
    masses = []
    current_seg_index = 0
    while i < doc_length:
        tokens_in_segment += 1
        if current_seg_index < len(segmentation) and i > segmentation[current_seg_index] - 1:
            masses.append(tokens_in_segment)
            tokens_in_segment = 0
            current_seg_index += 1
        i += 1
    masses.append(doc_length - segmentation[-1])

    return segeval.boundary_string_from_masses(tuple(masses))


# helper functions for language value in transcript json files
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
