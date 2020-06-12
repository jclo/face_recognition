# ******************************************************************************
"""
Short description.

Long description ...


Private Functions:
    . none,


Public:
    . none,


@namespace      _
@author         <author_name>
@since          0.0.0
@version        0.0.0
@licence        MIT. Copyright (c) 2020 Mobilabs <contact@mobilabs.fr>
"""
# ******************************************************************************
import numpy as np

LABEL_PATH = './datasets/labels_v1_8631.npy'


def decode_predictions(predictions, top=3):
    """Decodes the predictions returned by VGGFace2.

    ### Parameters:
        param1 (arr):       the resulting predictions.
        param2 (num):       the number of the most valuable predictions.

    ### Returns:
        (arr):              returns the matching people names,

    ### Raises:
        ValueError          if the predictions isn't a 2D array.
        ValueError          if the prediction doesn't contain 8631 values.
    """

    # predictions must be an array of array containing 8631 float numbers:
    # [[n0, n1, ... n8630]]
    if len(predictions.shape) == 2:
        if predictions.shape[1] == 8631:
            LABELS = np.load(LABEL_PATH)
        else:
            raise ValueError('decode_predictions expects an array with 8631 values!')

    else:
        raise ValueError('decode_predictions expects a 2D array!')

    # Extract the top predictions:
    results = []
    for pred in predictions:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
            # results.append([str(LABELS[i].encode('utf8')), pred[i]])
            results.append([str(LABELS[i]), pred[i]])

    return results

# -- o ----
