# lobe-python-imagesfromfolder2labeledFolder
Multiple image prediction + CSV output + sort copy of the images into labeled folders

What this Python script does:
# 01. Batch process a folder with JPG-images
# 02. Label each image based on your Lobe-trained CV model
# 02a - assign a label and confidence score (for that label)
# 03 Create separate folders based on the labels you used in your model
# 04 Copy every image to the labeled folder after classifying

For Mac users: before you run this script:
1. Create a folder called 'predictions': this is where the newly created/updated CSV-file goes.
2. make sure you have deleted the ".DS_Store" file in the 'imgs' folder
You can do this by running this command in your Terminal: rm .DS_Store

# You need this import in order to read the images from the folder AND to create the folders to add the labeled images to
	import os 

# to create the CSV file
	import csv 

# used for copying images to the designated predicted label-folders
	import shutil 

# this is the folder you have to create, where your testset images go
# you can change 'imgs' to any name you desire
path1 = "imgs"   


# to sort the files into their label-destination folders
def copyFilesToDestLabelFolders(sourceFile, destination):
    # Copy file to another directory
    newPath = shutil.copy(sourceFile, destination)
    print("Path of copied file : ", newPath)

# define the name of the directory to be created
def makeLabelDirs(whichLabel):
    print("whichLabel: " + whichLabel)
    realLabel = whichLabel.strip()
    try:
        os.mkdir(realLabel)
    except OSError:
        print ("Creation of the directory %s failed" % realLabel)
    else:
        print ("Successfully created the directory %s " % realLabel)


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



# the model you've trained using Lobe.ai and exported choosing 'TensorFlow, Use your model in a Python app'
	from lobe import ImageModel #the model you've trained using Lobe.ai and exported choosing 'TensorFlow, Use your model in a Python app'

	model = ImageModel.load('')

	def createPredictionsCSV(theoutcome):

		#'a' is for append
		with open('predictions/predictions.csv', 'a') as file:
			writer = csv.writer(file)
			writer.writerow(theoutcome);

# for opening the folder containing your images
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
		# This 'if' statement  writes the scores pnly for the predicted label 
		# it copies the sourcefiles from the original source-folder ('imgs')  to the newly created  & predicted label-named folder:
		# e.g. all images predicted as "label1" are copied from the source folinto the newly created folder "label1"
		if label is result.prediction:
		    print("im" + im)
		    copyFilesToDestLabelFolders(im, label)
		    predList = [pics, im, result.prediction, labelScore]
		    createPredictionsCSV(predList)
