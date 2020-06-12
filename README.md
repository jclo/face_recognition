# README

This project implements a technique for Face Recognition in Keras based on [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/). It recognizes the people that are referenced in the dataset (see the list [here](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta/identity_meta.csv)).

This [paper](https://arxiv.org/pdf/1710.08092.pdf) explains what is the dataset named `VGGFace2`.


## Quick Startup

### Download the network weights

You need to download the network `weights` [here](https://drive.google.com/file/d/1niCSaaqbXs6YP1B-DddRFQ7RAWsU8OW_/view?usp=sharing) and copy them in the `weights` folder.

### Download datasets

You can download a small dataset of images [here](https://drive.google.com/file/d/1m71-uiPA67NPPXqAfQwHVpdk79Ppj9vE/view?usp=sharing). Once, downloaded unzip the file in the `datasets` folder.

You also need to download the label file [here](https://drive.google.com/file/d/14-omxZDlz16zr1Xb15QkRfaPlu3NxSHA/view?usp=sharing). This file is used to convert the predictions of the `VGGFace` neural network to human readable information. Once, downloaded unzip the file in the `datasets` folder.

### Create a Virtual Environment

If you are on MacOS or Linux, type the following command to create a python virtual environment and install the required packages:

```bash
./configure.sh
```

Then, activate the virtual environment:

```bash
source .venv/bin/activate
```

***Nota***:
<br>This project requires `python 3`. You must install it first.


## Run

Execute the command:

```bash
python src/main.py -d 'dlib' -s './datasets/people/tim_cook.jpg'
```

You must get:

```bash
decoding ./datasets/people/tim_cook.jpg using ResNet50 and dlib face detector ...
Tim_Cook: 96.630%
Laurent_Ruquier: 0.258%
Mads_Gilbert: 0.069%
Rupert_Stadler: 0.061%
Jack_Layton: 0.048%
Done!
```

## How it works

The code is pretty easy to understand. The file `src/main.py` contains the class that performs the operations: decode face, predict and decode predictions.

```python
class Finder:
    def __init__(self):
        """Creates the VGGFace2 neural network, MTCNN and dlib face detectors."""
        self.vggface = ResNet50(n_classes=8631)
        self.vggface.load_weights(WEIGHTS)
        self.mtcnn = MTCNN()
        self.dlib = Dlib()

    def whois(self, image, detector, top=3):
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
```

The face is extracted from an image with `dlib` implemented in the file `src/decoders/dlib.py` or [MTCNN](https://github.com/ipazc/mtcnn).

The prediction is done through the `VGGFace` neural network implemented in the file `src/models/resnet50.py`.

Then, the predictions are decoded by the function `decode_predictions` implemented in the file `src/util/util.py`.


## License

MIT.
