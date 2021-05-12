# README

## Setting up the environment

This program requires ```pipenv``` to be installed beforehand, it can be installed easily with ```pip```

```
pip install pipenv
```

Then, all the dependencies can be installed using the following command:

```
pipenv install
```

With all the dependencies installed, run the environment

```
pipenv shell
```

You would need to download the model for ```dlib```, which is the file ```shape_predictor_68_face_landmarks.dat```
Download link : https://github.com/davisking/dlib-models

## Running the program

Simply run ```python main.py```.

The program will automatically generate frame morph sequences for each of the source-target pair in the folder ```img``` and place them in ```out1``` and```out2``` folders respectively.

For the annotator program , use```python annotator.py```, the resulting points will be stored in ```annotated_points.txt```, note that the points have to be hardcoded into the main program. By default, this opens the ```target2.png```.


