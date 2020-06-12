# ******************************************************************************
"""
Detects or extracts a face(s) from an image with the dlib decoder.


Private Functions:
    . none,


Public Class:
    .  Haar                     a class to detect or extract a face from an image,


Private Methods:
    . __detect_face             detects the face(s) on the passed-in image,


Public Methods:
    . get_face                  extracts the face from the passed-in image,
    . get_image_with_faces      detects and highlights the face(s) on the image,


@namespace      _
@author         <author_name>
@since          0.0.0
@version        0.0.0
@licence        MIT. Copyright (c) 2020 Mobilabs <contact@mobilabs.fr>
"""
# ******************************************************************************
import cv2
import dlib

RECT_COLOR = (0, 255, 255)
RECT_THICKNESS = 2


# -- Public --------------------------------------------------------------------

class Dlib:
    """A class to detect or extract a face from an image with dlib.

    ### Attributes:
        classifier (obj):       the dlib decoder object.

    ### Methods:
        get_face(image):
            extracts the face from the passed-in image.

        get_image_with_faces(image):
            detects and highlights the face(s) on the passed-in image.

    ### Raises:
        none
    """

    def __init__(self):
        """Creates the dlib decoder."""
        self.detector = dlib.get_frontal_face_detector()

    def __detect_face(self, img):
        """Detects the face(s) on the passed-in image.

        ### Parameters:
            param1 (str):   the path of the input image.

        ### Returns:
            (arr):          returns the coordinates of the detected face(s).

        ### Raises:
            -
        """
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        return self.detector(gray, 1)

    def get_face(self, image):
        """Extracts the face from the passed-in image.

        ### Parameters:
            param1 (arr):   the input image.

        ### Returns:
            (arr):          returns the extracted face.

        ### Raises:
            -
        """
        face = self.__detect_face(image)[0]
        x1, y1, x2, y2, _, _ = face.left(), face.top(), \
            face.right() + 1, face.bottom() + 1, face.width(), face.height()
        return image[y1:y1 + (y2 - y1), x1:x1 + (x2 - x1), :]

    def get_image_with_faces(self, image):
        """Detects and highlights the face(s) on the passed-in image.

        ### Parameters:
            param1 (arr):   the input image.

        ### Returns:
            (arr):          returns the input image with highlighted face(s).

        ### Raises:
            -
        """
        img = image.copy()
        faces = self.__detect_face(img)

        if len(faces) > 0:
            for i, d in enumerate(faces):
                x1, y1, x2, y2, _, _ = d.left(), d.top(), \
                    d.right() + 1, d.bottom() + 1, d.width(), d.height()
                cv2.rectangle(img, (x1, y1), (x2, y2), RECT_COLOR, RECT_THICKNESS)
        return img

# -- o ---
