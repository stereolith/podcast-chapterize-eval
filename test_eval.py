import pytest, sys, inspect
sys.path.append('./podcast_chapterize')

def test_transcript_parse():
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