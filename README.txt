
Persona 3 Reload Date Recogniser

This project uses Python generate synthetic images of the "date" icon on the top right of the P3R screen.
It can generate a database of those synthetic images, and train a NN model to recognise the "date" images, 
both synthetic, and from the game.

The Python libaries used are:
- pytorch
- pillow
- matplotlib
- random

The architecture of the model is CNN (Convolutional Neural Network). Training on pixel and string information in .npz format from own database.

How to Use

Open your terminal after saving the desired file and download the needed libraries,
by running command "pip install -r requirements.txt".

For generating databases, go to "2datasetgen.py" and select how many images you want to generate
for the database, and run the file.

For training the model, go to "3modelgen.py" and adjust the hyperparameters if you please, then
run the file and train the model.

Open your terminal after saving the desired file and download the needed libraries,
by running command "pip install -r requirements.txt".

then either generate a synthetic image, 
through running "1imagegen.py" or add an image from the game, rename to "test_image" and run 
the recogniser.

Known Issues
- It is highly inaccurate.

Possible Solutions
- Change hyperparameters.
