# script to automatically evaluate different sets of parameters in the chapterization process

import os, sys, json, segeval
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

# change sys path to allow realtive imports of podcast-chapterize modules
import sys
from podcast_chapterize.chapterize.preprocessor_helper import lemma


def main():
    """
        main evaluation script. Fetches transcripts, creates parameter matrix, runs evaluation on all transcripts/ parameter sets
        and saves results to json file
    """    
    transcripts = get_transcripts()


    test_values = {
        'window_width': [50, 100, 200, 300],
        'with_boundaries': [0],
        'tfidf_min_df': [0, 5],
        'tfidf_max_df': [0.7, 0.9],
        'savgol_params': [
            {
                'savgol_window_length': 5,
                'savgol_polyorder': 3        
            },
            {
                'savgol_window_length': 7,
                'savgol_polyorder': 4        
            },
            {
                'savgol_window_length': 9,
                'savgol_polyorder': 6        
            },
        ],
        'doc_vectorizer': ['tfidf', 'ft_sif_average', 'ft_average', 'ft_sum']
    }

    param_matrix = parameter_matrix(test_values)

    score_matrix = run_evaluation(transcripts, param_matrix)

    results_dict = {
        'parameter_matrix': [param.to_dict() for param in param_matrix],
        'test_values': test_values,
        'transcripts': [transcript['filename'] for transcript in transcripts],
        'results': score_matrix
    }

    with open('results.json', 'w') as f:
        json.dump(results_dict, f)

def run_evaluation(transcripts, param_matrix):
    """
        runs chapterization process on transcrips with the given parameter sets and evaluates segmentation results compared to
        the given gold standard segmentations.

    Args:
        transcripts (list): transcripts with gold standard segmentations to test parameter sets against
        param_matrix (list of ChapterizerParameter): parameter sets to test

    Returns:
        [type]: [description]
    """    
    from multiprocessing import Pool
    from functools import partial
    from contextlib import closing

    eval_score_matrix = [[] for i, x in enumerate(param_matrix)] # transcript scores by parameter set

    for i, transcript in enumerate(transcripts):
        # use multiprocesses to distribute over multiple cores
        with closing( Pool() ) as pool:
            get_score_for_current_transcript = partial(get_score, transcript=transcript, true_chapter_boundaries=transcript['trueChapterBoundaries'], i_transcript=i, len_transcripts=len(transcripts), len_params=len(param_matrix))
            scores = pool.map(get_score_for_current_transcript, enumerate(param_matrix))
            print(scores)

        for j, score in enumerate(scores):
            eval_score_matrix[j].append(float(score))

    return eval_score_matrix

    

def get_score(params_tuple, transcript, true_chapter_boundaries, i_transcript, len_transcripts, len_params):
    """
        Calculate a segmentation score for a given segmentation, compared to the gold standard segmentation.
        Returns a 0 score and skips segeval calculation if gold standard or tested segmentation has < 2 boundaries.

    Args:
        params_tuple (tuple): ChapterizerParameter and index of current parameter configuration (handled as tuple to allow execution from pool.map function)
        transcript (obj): underlying transcript for the segmentation
        true_chapter_boundaries (list): gold standard segmentation
        i_transcript (int): index of current transcript
        len_transcripts (int): total number of transcripts
        len_params (int): total number of parameter configutations to test (no of rows of parameter matrix)

    Returns:
        float: segment evaluation score
    """    
    params = params_tuple[1]
    i_params = params_tuple[0]
    with nostdout():
        boundaries = run_chapterization(transcript, params)
        if len(boundaries) < 2 or len(true_chapter_boundaries) < 2:
            score = 0
        else:
            try:
                score = eval_segmentation(boundaries, true_chapter_boundaries, len(transcript['tokens']))
            except:
                print('evaluation of segmentation failed, setting score to 0')
                score = 0
    print(f'tested transcript {i_transcript + 1}/{len_transcripts} with parameters {i_params + 1}/{len_params}\n  found boundaries: {len(boundaries)}, gold boundaries: {len(true_chapter_boundaries)}')
    return score


def get_transcripts():
    """get pickeled transcripts that were prepared by the prepare_transcripts function

    Returns:
        list: list of transcripts
    """
    import pickle
    return pickle.load(open('transcripts/transcripts.pickle', 'rb'))

def parse_transcripts_json():
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
            transcript['filename'] = f.name
        try:
            l = transcript['language']
            c = transcript['chapters']
            transcript['tokens'] = [TranscriptToken.from_dict(token) for token in transcript['tokens']]
            transcripts.append(transcript)
        except:
            print(f'skipping transcript {file} because it is missing language and/or chapter values')

    print(f'fetched {len(transcripts)} transcripts')

    return transcripts


class ChapterizerParameter:
    """class that holds parameter configuration for the chapterizer
    """
    def __init__(
        self,
        window_width,
        with_boundaries,
        tfidf_min_df,
        tfidf_max_df,
        savgol_params,
        doc_vectorizer
    ):
        self.window_width = window_width
        self.with_boundaries = with_boundaries
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.savgol_params = savgol_params
        self.doc_vectorizer = doc_vectorizer
    
    def to_dict(self):
        return {
            'window_width': self.window_width,
            'with_boundaries': self.with_boundaries,
            'tfidf_min_df': self.tfidf_min_df,
            'tfidf_max_df': self.tfidf_max_df,
            'savgol_params': self.savgol_params,
            'doc_vectorizer': self.doc_vectorizer
        }

def parameter_matrix(test_values):
    """create a parameter matrix (list of parameter dicts) for every combination of possible parameters
    
    Returns:
        [list of ChapterizerParameter]: parameter configurations
    """

    import itertools
    param_matrix = list(itertools.product(*[
        test_values['window_width'],
        test_values['with_boundaries'],
        test_values['tfidf_min_df'],
        test_values['tfidf_max_df'],
        test_values['savgol_params'],
        test_values['doc_vectorizer']
    ]))

    param_objects = [ChapterizerParameter(*params) for params in param_matrix]

    return param_objects


def run_chapterization(transcript, params):
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
        savgol_window_length=params.savgol_params['savgol_window_length'],
        savgol_polyorder=params.savgol_params['savgol_polyorder'],
        doc_vectorizer=params.doc_vectorizer
    )

    try:
        boundaries = c.chapterize(
            transcript['tokens'],
            boundaries=[],
            language=transcript['language'],
            skip_lemmatization=True
        )
    except:
        print('chapterize failed')
        boundaries = [0]

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

def results(results_path):
    """prints results: top 10 scoring parameter configurations

    Args:
        results_path (str): path to results.json, a file generated by the main() function
    """    
    import json, pprint
    import numpy as np
    pp = pprint.PrettyPrinter(indent=4)

    with open(results_path) as f:
        results = json.load(f)

    mean_scores = []
    for res in results['results']:
        mean_scores.append(np.mean(res))

    sorted_sums_i = numpy.argsort(mean_scores)

    for i in sorted_sums_i[::-1][:10]:
        print(f'segeval mean score: {mean_scores[i]}')
        pp.pprint(results['parameter_matrix'][i])
        print(f'\n')

def plot_results(results_path):
    """plot mean score values for every tested parameter value

    Args:
        results_path (str): path to results.json, a file generated by the main() function
    """    
    import json, pprint
    import numpy as np
    import matplotlib.pyplot as plt

    pp = pprint.PrettyPrinter(indent=4)

    with open(results_path) as f:
        results = json.load(f)

    mean_scores = []
    for res in results['results']:
        mean_scores.append(np.mean(res))

    param_value_avgs = {}

    # mean score for values for parameters
    scores_by_param_value = {}
    for param in results['test_values']:
        scores_by_param_value[param] = {}
        for value in results['test_values'][param]:
            scores_by_param_value[param][str(value)] = []

    for i, param_set in enumerate(results['parameter_matrix']):
        for param in param_set:
            scores_by_param_value[param][str(param_set[param])] += results['results'][i]

    # calculate mean scores for any given parameter value
    mean_score_by_param_value = scores_by_param_value
    for param in scores_by_param_value:
        for param_value in scores_by_param_value[param]:
            mean_score_by_param_value[param][param_value] = np.mean(scores_by_param_value[param][param_value])
    
    # plot mean scores for parameter value
    for param in mean_score_by_param_value:
        width = 0.35
        labels = list(mean_score_by_param_value[param].keys())
        scores = list(mean_score_by_param_value[param].values())
        fig, ax = plt.subplots()

        ax.bar(labels, scores, width)

        ax.set_ylabel('mean score')
        ax.set_title(f'Mean segmentation evaluation scores for parameter {param}')
        ax.legend()

        plt.show()

    

    # for param in results['parameter_matrix'][0]:
    #     for i, param_set in enumerate(results['parameter_matrix']):
    #         param_value_avgs[param] = 


if __name__ == "__main__":
    main()
