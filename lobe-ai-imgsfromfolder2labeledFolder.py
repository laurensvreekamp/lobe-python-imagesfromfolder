###########################################################################
# This Lobe.ai-Python TensorFlow-script let's you do a few things:
# 
# 01. Batch process a folder with JPG-images
# 02. Label each image based on your Lobe-trained CV model
# 02a - assign a label and confidence score (for that label)
# 03 Create separate folders based on the labels you used in your model
# 04 Copy every image to the labeled folder after classifying
# 05 Create a CSV-file in the 'predictions' folder, with a new row
#    for the entire batch of images in the 'imgs'folder, containing per image item / row this data:
#    - Path to image file
#    - Imagefilename
#    - Assigned label
#    - Confidence score



### WARNING ###
# For Mac users: before you run this script,
# make sure you have deleted the ".DS_Store" file in the 'imgs' folder
# You can do this by running this command in your Terminal:

# rm .DS_Store

##############

import os # You need this import in order to read the images from the folder AND to create the folders to add the labeled images to

import shutil # used for copying images to the designated predicted label-folders

import csv #for creating the predictions CSV file

from lobe import ImageModel #the model you've trained using Lobe.ai and exported choosing 'TensorFlow, Use your model in a Python app'

# create a folder for all your results
os.mkdir('results')

# this is the folder you have to create, where your testset images go
# you can change 'imgs' to any name you desire
path1 = "imgs"   

# to sort the files into their label-destination folders
def copyFilesToDestLabelFolders(sourceFile, destination):
    # Copy file to another directory
    newPath = shutil.copy(sourceFile, "./results/" + destination)
    print("Path of copied file : ", newPath)

# define the name of the directory to be created
def makeLabelDirs(whichLabel):
    print("whichLabel: " + whichLabel)
    realLabel = "./results/" + whichLabel.strip()
    try:
        os.mkdir(realLabel)
    except OSError:
        print ("Creation of the directory %s failed" % realLabel)
    else:
        print ("Successfully created the directory %s " % realLabel)

# function to check if there are empty files and if it even does exist
def is_file_empty(file_path):
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0

# Every Lobe-project generates a 'labels.txt' file
# You can read the txt file and create folders corresponding to the labelnames in that txt-file
def sortImagesToLabeledFolder():
    lines = []
    labelnames = []
    with open('labels.txt', encoding='utf8') as l:
        lines = l.readlines()

    count = 0
    for line in lines:
        count += 1
        #print(f'line {count}: {line}')
        labelnames.append(line)
        makeLabelDirs(line)

sortImagesToLabeledFolder()



############################################################################################
# this is where the lobe.ai Model will be put to use to predict a label and assign a score #
#############################################################################################
model = ImageModel.load('')

# create the CSV file to write the image- & prediction-data into
with open('./results/predictions.csv', 'w') as file:
	writer = csv.writer(file)
	writer.writerow(["imgId", "imagePath", "labeledAs", "confidenceScore"])

def createPredictionsCSV(theoutcome):

	#'a' is for append
	with open('./results/predictions.csv', 'a') as file:
		writer = csv.writer(file)
		writer.writerow(theoutcome);

# for opening the folder containing your images, and filter out everything that isn't jpg
listing = list(filter(lambda f: f.endswith('.jpg'), os.listdir(path1)))

for pics in listing:
    im = path1 + '/' + pics
    
    # Check if file is empty
    is_empty = is_file_empty(im)
    
    if is_empty:
        pass
    else:
        result = model.predict_from_file(im)

        predList = [pics, im, result.prediction]

        for label, confidence in result.labels:
            labelScores = f"{label}: {confidence*100}%"
            print(f"{label}: {confidence*100}%")
            labelName = (f"{label}")
            labelScore = (f"{confidence*100}")

            # Since I'm fairly new to Python, I haven't found a way to add all scores for each label to an image.
            # This 'if' statement  writes the scores pnly for the predicted label 
            # it copies the sourcefiles from the original source-folder ('imgs')  to the newly created  & predicted label-named folder:
            # e.g. all images predicted as "label1" are copied from the source folder into the newly created folder "label1"
            if label is result.prediction:
                print("im" + im)
                copyFilesToDestLabelFolders(im, label)
                predList = [pics, im, result.prediction, labelScore, labelScore[2]]
                createPredictionsCSV(predList)

