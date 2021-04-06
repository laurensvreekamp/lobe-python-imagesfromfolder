# lobe-python-imagesfromfolder
Multiple image prediction + CSV output

What this Python script does:
 - it reads all images (only JPG-file format) from a specified folder, 
 - classifies & scores each image based on your Lobe-trained CV model, 
 - creates a CSV file 
 - adds a row for each image, containing the prediction data for that image

For Mac users: before you run this script,
make sure you have deleted the ".DS_Store" file in the 'imgs' folder
You can do this by running this command in your Terminal:
rm .DS_Store

# Here's the script:

import os #You need this import in order to read the images from the folder
import csv #to create the CSV file
from lobe import ImageModel #the model you've trained using Lobe.ai and exported choosing 'TensorFlow, Use your model in a Python app'

model = ImageModel.load('')

# create the CSV file to write the image- & prediction-dat into
with open('predictions/predictions.csv', 'w') as file:
	writer = csv.writer(file)
	writer.writerow(["imgId", "imagePath", "labeledAs", "confidenceScore"])

def createPredictionsCSV(theoutcome):

	#'a' is for append
	with open('predictions/predictions.csv', 'a') as file:
		writer = csv.writer(file)
		writer.writerow(theoutcome);
	
path1 = "imgs"   

# for opening the folder containing your images
listing = os.listdir(path1)  

for pics in listing:
    im = path1 + '/' + pics
    result = model.predict_from_file(im)

    predList = [pics, im, result.prediction]

    for label, confidence in result.labels:
    	labelScores = f"{label}: {confidence*100}%"
    	print(f"{label}: {confidence*100}%")
    	labelName = (f"{label}")
    	labelScore = (f"{confidence*100}")

    	# Since I'm fairly new to Python, I haven't found a way to add all scores for each label to an image.
    	# This 'if' statement only writes the scores for the predicted the label 
    	if label is result.prediction:
    		predList = [pics, im, result.prediction, labelScore]
    		createPredictionsCSV(predList)
