import pytest, sys, inspect
sys.path.append('./podcast_chapterize')

def test_transcript_fetch():
    from eval_chapterization import get_transcripts
    import importlib  

    from podcast_chapterize.transcribe.SpeechToTextModules.SpeechToTextModule import TranscriptToken
    transcripts = get_transcripts()

    assert len(transcripts) > 1
    for transcript in transcripts:
        assert isinstance(transcript['tokens'][0], TranscriptToken)

def test_parameter_matrix():
    from eval_chapterization import parameter_matrix, ChapterizerParameter

    p_matrix = parameter_matrix()

    for parameter_vector in p_matrix:
        assert isinstance(parameter_vector, ChapterizerParameter)

def test_eval_segmentations():
    # test data: 1 exact match, 3 near matches (less than 100 tokens apart), rest is not matched
    sample_segmentation = [200, 1200, 2000, 3000, 5400, 6200, 7400, 8200, 9200, 10060, 10800, 12200, 13200, 13800, 16000, 18400, 20000, 21200, 22400, 23200]
    sample_segmentation_gold = [1, 76, 922, 1751, 2950, 3230, 3950, 4991, 6504, 9202, 10070, 13200, 16236, 21929, 22560, 24404]
    sample_doc_length = 24800

    from eval_chapterization import eval_segmentation

    score = eval_segmentation(sample_segmentation, sample_segmentation_gold, sample_doc_length)
    assert score > 0

def test_run_chapterization_bad_parameters():
    from eval_chapterization import ChapterizerParameter, get_transcripts, run_chapterization

    bad_params = ChapterizerParameter(
        0,
        False,
        None,
        -122,
        {
            'savgol_window_length': -9.5,
            'savgol_polyorder': 20        
        },
        None
    )

    transcript = get_transcripts()[0]

    chapters = run_chapterization(transcript, bad_params)

    assert chapters == [0]



# integration test: evaluation on a subset of available transcripts
def test_evaluation():
    from eval_chapterization import run_evaluation, get_transcripts, parameter_matrix

    transcripts = get_transcripts()[0:1]

    param_matrix = parameter_matrix()

    results = run_evaluation(transcripts, param_matrix)

    assert len(results) == len(param_matrix) and len(results[0]) == len(transcripts)