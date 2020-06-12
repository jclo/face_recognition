# ******************************************************************************
"""
Finds people on the passed-in image.

Implements VGGFace2 on keras for performing people face recognition as described
here: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.


Private Functions:
    . _parse                    parses the script arguments.
    . _mtcnn_extract_face       extracts a face using the MTCNN face detector.
    . _dlib_extract_face        extracts a face using the dlib face detector.


Public Class:
    .  Finder                   a class to find people on an image.


Public Methods:
    . whois                     finds who is the person on the image.


@namespace      _
@author         <author_name>
@since          0.0.0
@version        0.0.0
@licence        MIT. Copyright (c) 2020 Mobilabs <contact@mobilabs.fr>
"""
# ******************************************************************************
import argparse
from decoders.dlib import Dlib
from models.resnet50 import ResNet50
from util.util import decode_predictions

import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

WEIGHTS = './weights/vggface_tf_resnet50_v1_8631.h5'


# -- Private Functions ---------------------------------------------------------

def _parse():
    """Parses the script arguments.

    ### Parameters:
        param1 ():          none,

    ### Returns:
        (str):              returns the option values,

    ### Raises:
        none
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, help='path of the input image')
    parser.add_argument('-d', '--decoder', type=str, help='the face decoder mtcnn or dlib')
    args = parser.parse_args()

    if args.source is None or args.decoder is None:
        print('You must provide the path of the image to analyze and the decoder!')
        parser.print_help()
        exit()
    else:
        return args.source, args.decoder


def _mtcnn_extract_face(detector, image, required_size=(224, 224)):
    """Extracts a face from an image using the MTCNN face detector.

    ### Parameters:
        param1 (obj):       the MTCNN detector object.
        param2 (arr):       the input image.
        param3 (tuple):     the face size.

    ### Returns:
        (arr):              returns the detected face.

    ### Raises:
        none
    """
    # detect faces in the image and extract the bounding box
    # from the first face
    results = detector.detect_faces(image)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]

    # resize the extracted face to the size required by
    # the VGGFace CNN
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def _dlib_extract_face(detector, image, required_size=(224, 224)):
    """Extracts a face from an image using the dlib face detector.

    ### Parameters:
        param1 (obj):       the Dlib detector object.
        param2 (arr):       the input image.
        param3 (tuple):     the face size.

    ### Returns:
        (arr):              returns the detected face.

    ### Raises:
        none
    """
    face = detector.get_face(image)
    face = cv2.resize(face, required_size, interpolation=cv2.INTER_LINEAR)
    return face


# -- Public --------------------------------------------------------------------

class Finder:
    """Class for a Face Recognition.

    ### Attributes:
        vggface (obj):      VGGFace2 Neural network.
        mtcnn (obj):        MTCNN face detector object.
        dlib (obj):         Dlib face detector object.

    ### Methods:
        whois(image, detector):
            finds who is the person on the image.

    ### Raises:
        Exception           if the detector isn't supported.
        """

    def __init__(self):
        """Creates the VGGFace2 neural network, MTCNN and dlib face detectors."""
        self.vggface = ResNet50(n_classes=8631)
        self.vggface.load_weights(WEIGHTS)
        self.mtcnn = MTCNN()
        self.dlib = Dlib()

    def whois(self, image, detector, top=3):
        """Finds who is the person on the image.

        ### Parameters:
            param1 (arr):   the input image.
            param2 (obj):   the face detector to use.
            param3 (num):   the number of the most valuable predictions.

        ### Returns:
            (arr):          returns the most valuable predictions.

        ### Raises:
            none
        """

        # First, extract the face from the passed-in image with 'mtcnn'
        # or 'dlib' face detector.
        if detector == 'mtcnn':
            face = _mtcnn_extract_face(self.mtcnn, image)
        elif detector == 'dlib':
            face = _dlib_extract_face(self.dlib, image)
        else:
            raise Exception('Sorry, only mtcnn and dlib decoders are allowed!')

        # Then, convert the face image from (224, 224, 3) to four dimensions,
        # and the amplitude from 0 - 255 to -127.5 - +127.5. Finally, perform
        # the predictions and convert these predictions to an human readable
        # information.
        x = face - 127.5
        image = np.expand_dims(x, axis=0)
        preds = self.vggface.predict(image)
        results = decode_predictions(preds, top)
        return results


if __name__ == '__main__':
    source, detector = _parse()
    finder = Finder()

    if detector == 'mtcnn':
        print('decoding ' + source + ' using ResNet50 and mtcnn face detector ...')
        image = pyplot.imread(source)
        results = finder.whois(image, 'mtcnn', top=5)

    elif detector == 'dlib':
        print('decoding ' + source + ' using ResNet50 and dlib face detector ...')
        image = cv2.imread(source)
        results = finder.whois(image, 'dlib', top=5)

    else:
        print('The face detector must be mtcnn or dlib')
        exit()

    # display the most likely results
    for result in results:
        print('%s: %.3f%%' % (result[0].lstrip(), result[1] * 100))
    print('Done!')

# -- o ----
