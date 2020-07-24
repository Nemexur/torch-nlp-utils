from typing import List
import torch


def pad_3d_sequence(
    tokens: List[List[int]],
    max_sent_length: int = None,
    max_sents: int = None
) -> torch.Tensor:
    """
    Perform padding for 3D Tensor
    (Batch Size x Number of sentences x Number of words in sentences).

    Parameters
    ----------
    tokens : `List[List[int]]`, required
        Nested lists of token indexes with variable
        number of sentences and number of words in sentence.
    max_sent_length : `int`, optional (default = `None`)
        Max number of words in sentence.
        If `None` number of words in sentence
        would be determined from passed `tokens`
        (equals max number of words in sentence per batch).
    max_sents : `int`, optional (default = `None`)
        Max number of sentences in one document.
        If `None` max number of sentences
        would be determined from passed `tokens`
        (equals max number of sentences per batch).

    Returns
    -------
    `torch.Tensor`
        Padded 3D torch.Tensor.

    Examples:
    ---------
        pad_3d_sequence(
            [[[1, 2, 3], [4, 5]], [[3, 4], [7, 8, 9, 6], [1, 2, 3]]]
        )
        tensor([[[1., 2., 3., 0.],
                 [4., 5., 0., 0.],
                 [0., 0., 0., 0.]],

                [[3., 4., 0., 0.],
                 [7., 8., 9., 6.],
                 [1., 2., 3., 0.]]])
    """
    # Adopted from: https://discuss.pytorch.org/t/nested-list-of-variable-length-to-a-tensor/38699
    words = max_sent_length if max_sent_length else max([len(row) for batch in tokens for row in batch])
    sentences = max_sents if max_sents else max([len(batch) for batch in tokens])
    padded = [batch + [[0] * (words)] * (sentences - len(batch)) for batch in tokens]
    padded = torch.Tensor([row + [0] * (words - len(row)) for batch in padded for row in batch])
    padded = padded.view(-1, sentences, words)
    return padded
