### Training the classifier

The classifier used in this project was a 2 hidden layered neural network with 400 neurons, and trained for 500 iterations. The average running time on i7 processor 16GB RAM took around 7-9 minutes.

Attached to the folder is file named train.py. To run the script simply type in the terminal `python train.py` and it will save the trained model in file named "neural_model.sav". After forming the preprocessing that is described below, put the path of the folder of images in the variable path_to_dataset in the code.

The dataset used is the [HOMUS](https://grfia.dlsi.ua.es/homus/) dataset, but we applied some preprocessing on this dataset. The dataset comes natively in a '.txt' format so it must be converted to imgs.

In order to create the dataset correctly the follwing steps must be done:
1. Download dataset from https://grfia.dlsi.ua.es/homus/
2. Merge the dataset folders into one folder
3. Convert the '.txt' files to images, by drawing lines between points that are given in the '.txt' files
4. Make the image object fit to the borders of the image(as if zooming into the image till it contains the whole space)
5. Delete all images that are related to Barline, common time, cut time, 2-2, 3-4, 3-8, 6,8, 9-8, 12-8,C-clef, F-clef, as they are not required to be classified in the project
