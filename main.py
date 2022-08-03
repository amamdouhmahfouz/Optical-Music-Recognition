
import argparse
import os
import datetime
from omr import *
# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("inputfolder", help = "Input File")
parser.add_argument("outputfolder", help = "Output File")

args = parser.parse_args()

if not sys.warnoptions:
    warnings.simplefilter("ignore")
t = time.time()
for _,_,files in os.walk(args.inputfolder):
    for file in files:
        if ".jpg" in file.lower() or ".jpeg" in file.lower() or "png" in file.lower() or "bmp" in file.lower():
            sheet = SheetMusic(args.inputfolder+"/"+file, 'nn86_model.sav')
            filename = file.split('.')[0] + ".txt"
            try:
                score = sheet.run()
            except:
                score = "[]"
            #print(score)
            with open(args.outputfolder+"/"+filename, "w") as text_file:
                text_file.write("%s" % score)


#print('program finished in: ',time.time()-t," seconds")
