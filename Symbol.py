import cv2
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,binary_opening, skeletonize, thin
from common import *
from BoundingBox import *
from Line import *

class Symbol:
    def __init__(self, img_symbol, x1, y1, x2, y2, img_bin, img_gray):
        self.position_label = None #one of the following: [c,d,e,f,g,a,b,c2,d2,e2,f2,g2,a2,b2]
        self.row = self.get_centroid(x1,y1,x2,y2)[1] #y position, will be used to get the position label
        self.col = self.get_centroid(x1,y1,x2,y2)[0] #x position, will be used to order the symbols on the lines
        self.img_bin = img_bin #binary image of the symbol
        self.img = img_symbol
        self.img_gray = img_gray
        self.classification = None # 'G-Clef', 'Quarter-Note', ... etc
        self.features = None # example: hog features
        self.target_img_size = (32, 32)

        # x1,y1,x2,y2 are absolute paths, i.e coordinates in the original whole image
        self.x1 = x1 #col1
        self.y1 = y1 #row1
        self.x2 = x2 #col2
        self.y2 = y2 #row2

        #if classification is wrong
        self.doubt = False

        self.quarters_pos = []
        self.keypoints = [] #better work with keypoint than quarters_pos
        self.base_row = None

        #using tempalate matchign
        self.quarters_bboxes = [] #bounding boxes of quarters only
        self.halfs_bboxes = [] #bounding boxes of halfs only

    def get_centroid(self,x1,y1,x2,y2):
        x_mid = int( (x1 + x2)/2 )
        y_mid = int( (y1 + y2)/2 )
        return x_mid, y_mid

    def extract_hog_features(self):
        """
        Extracts Histogram Of Features from input image
        """

        if self.img.max() <= 1:
            self.img = (self.img*255).astype('uint8')

        img = cv2.resize(self.img, self.target_img_size)
        win_size = (32, 32)
        cell_size = (4, 4)
        block_size_in_cells = (2, 2)

        block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
        block_stride = (cell_size[1], cell_size[0])
        nbins = 9  # Number of orientation bins
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        h = hog.compute(img)
        h = h.flatten()
        self.features = h
        return


    def classify_symbol(self, model):
        '''
        classify the type of the symbol, i.e 'G-Clef', 'Quarter-Note', ... etc
        '''

        #first extract the features
        self.extract_hog_features()

        if self.img.max() > 1:
            self.img = self.img / 255
            self.img= self.img.astype('uint8')
        imnew = self.img.copy()
        imnew = thin(imnew,5) # TODO: see the most suitable for all casses
        self.classification = model.predict([self.features])

        return



    def fit_quarters_bounding_boxes(self, bbox_lst):
        '''
        fits the bounding boxes list inside the symbol of possible
        input:
            - bbox_lst: list of BoundingBoxs with no duplicates
        '''
        if len(bbox_lst) == 0:
            return

        y1 = self.y1 - self.base_row
        y2 = self.y2 - self.base_row

        bboxes = bbox_lst.copy()
        bboxes.sort(key = lambda b: b.center[0], reverse=False) #sort ascendingly on center cols, sorts in-place
        for i in range(len(bboxes)):
            if bboxes[i].center[0] >= self.x1 and bboxes[i].center[0] <= self.x2 and bboxes[i].center[1] >= y1 and bboxes[i].center[1] <= y2:
                self.quarters_bboxes.append(bboxes[i])

        return

    def fit_halfs_bounding_boxes(self, bbox_lst):
        '''
        fits the bounding boxes list inside the symbol of possible
        input:
            - bbox_lst: list of BoundingBoxs with no duplicates
        '''
        if len(bbox_lst) == 0:
            return

        y1 = self.y1 - self.base_row
        y2 = self.y2 - self.base_row

        bboxes = bbox_lst.copy()
        bboxes.sort(key = lambda b: b.center[0], reverse=False) #sort ascendingly on center cols, sorts in-place

        for i in range(len(bboxes)):
            if bboxes[i].center[0] >= self.x1 and bboxes[i].center[0] <= self.x2 and bboxes[i].center[1] >= y1 and bboxes[i].center[1] <= y2:
                self.halfs_bboxes.append(bboxes[i])

        return


    def fit_quarters_and_keypoints(self, _quarters_pos, _keypoints):
        '''
        input:
            - _quarters_pos: ordered list of (row,col) center positions of quarters if any
            - _keypoints: a keypoint has quarter position(pt) and size
            NOTE: positions relative to the the belonging staff_group

        '''
        if len(_quarters_pos) == 0 or len(_keypoints) == 0:
            return

        quarters_pos = _quarters_pos.copy()
        keypoints = _keypoints.copy()

        # TODO: check this again
        y1 = self.y1 - self.base_row
        y2 = self.y2 - self.base_row

        for i in range(len(quarters_pos)):
            if quarters_pos[i][1] >= self.x1 and quarters_pos[i][1] <= self.x2 and quarters_pos[i][0] >= y1 and quarters_pos[i][0] <= y2:
                self.quarters_pos.append(quarters_pos[i])
                self.keypoints.append(keypoints[i])



        return


    def score(self, staff_labels_pos):
        '''
        input:
            - staff_labels_pos: a dictionary with key being the label of line(b2,a2,c2,..)
                                and value of row position taken from its belonging staff group
            NOTE: staff_labels_pos are coordinates with respect to the belonging staff group only
                  NOT the whole image

        output:
            - classification_string: examples: "a1/8", "{a1,b1,c2}", ...

        '''
        if "time" in self.classification[0].lower():
            if self.x1 > 500: #if col position is more than half the width of the image, then it is a misclassification of time
                return ""
            else:
                return symbols_labels[self.classification[0]]

        if self.classification[0] == "G-Clef" or self.classification[0] == "Barline": #no labels for it
            return ""

        classification_string = ""


        #sort staff_labels_pos in ascending order to start from the top position first
        staff_labels_pos_cpy = staff_labels_pos.copy()
        staff_labels_pos_cpy = {k: v for k, v in sorted(staff_labels_pos_cpy.items(), key=lambda item: item[1], reverse=False)}
        #separate both labels and row positions in separate lists
        staff_labels = [y[0] for y in staff_labels_pos_cpy.items()] #list of strings
        staff_pos = [y[1] for y in staff_labels_pos_cpy.items()] #list of numbers


        if len(self.quarters_bboxes) == 1:
            #only one quarter(full notehead) found
            quarter_row_pos = self.quarters_bboxes[0].center[1]
            nearest_line_idx = (np.abs(np.asarray(staff_pos) - quarter_row_pos)).argmin()
            staff_label = staff_labels[nearest_line_idx] #a string, i.e "a1","b1",....
            classification_string = staff_label #a string, i.e "a1","b1",....

            #we are certain that there is one quarter in the symbol
            if 'note' in self.classification[0].lower() and 'half' not in self.classification[0].lower():
                #if type is note, then we are sure of our classification, just need to detect position with respect to the line
                classification_string += symbols_labels[self.classification[0]] #this is the ending part of the string
            else: #our classification maybe wrong, due to the machine learning model
                #then we will believe the detect circles, and it will be a quarter note
                classification_string += "/4" #this is the ending part of the string

            #show_images([self.img_gray],[classification_string])
            return classification_string

        elif len(self.quarters_bboxes) > 1:
            #we are certain that there are multiple quarters(full noteheads)
            # We have 2 possibilities, either beaming or chords
            # 1. check for chords
            # 2. check for beams
            vertical_counts = 1   #for chords
            horizontal_counts = 1 #for beams
            prev_row = self.quarters_bboxes[0].center[1] # gets the center row position of the bounding box
            prev_col = self.quarters_bboxes[0].center[0]
            for i in range(1,len(self.quarters_bboxes)):
                #check if in approximately same column, note that it may be chords even if not exactly on top of each other, maybe back in back
                #if abs(self.quarters_bboxes[i].center[1] - prev_row) > 15: # TODO:change 18 to a suitable difference if any, but this is good till now
                if abs(self.quarters_bboxes[i].center[0] - prev_col) < 15:
                    vertical_counts +=1
                else: #they are beside each other
                    horizontal_counts += 1


            #see which one do we have chords or beams
            if vertical_counts > horizontal_counts:
                #duration_label example: "/1","/2","/4","/8",...
                #duration_label for chords is always ""/4"
                duration_label = "/4"

                classification_string = "{"
                tmp_str = "" #used for sorting alphabetically
                #Chords!
                #Check the lines that centers of quarters are near to
                for i in range(len(self.quarters_bboxes)):
                    quarter_row_pos = self.quarters_bboxes[i].center[1] #get row position
                    nearest_line_idx = (np.abs(np.asarray(staff_pos) - quarter_row_pos)).argmin()
                    staff_label = staff_labels[nearest_line_idx] #a string, i.e "a1","b1",....
                    if i == len(self.quarters_bboxes)-1:
                        classification_string += staff_label+duration_label+"}"
                    else:
                        classification_string += staff_label+duration_label+","

                #show_images([self.img_gray],[classification_string])
                return classification_string
            else:
                #beams!

                #use hough transform to detect nearly horizontal lines
                im = skeletonize(self.img)*255
                img_bin = img_as_ubyte(rgb2gray(im))

                # Detecting staff lines to get the angle of rotation
                tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
                h, theta, d = hough_line(img_bin, theta=tested_angles)

                # Generating figure 1
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                ax = axes.ravel()

                ax[0].imshow(img_bin, cmap=cm.gray)
                ax[0].set_title('Input image')
                ax[0].set_axis_off()



                ax[1].imshow(img_bin, cmap=cm.gray)
                origin = np.array((0, img_bin.shape[1]))
                hspace, angles, dists = hough_line_peaks(h, theta, d)

                ###### Remove outliers ########
                ## can remove using histograms, where we see the most angles we have
                # i.e removing false lines
                # mean = np.mean(angles)
                # standard_deviation = np.std(angles)
                # distance_from_mean = abs(angles - mean)
                # max_deviations = 2
                # not_outlier = distance_from_mean < max_deviations * standard_deviation
                # if standard_deviation > 0.08: #remove outliers
                #     angles = angles[not_outlier]
                #     hspace = hspace[not_outlier]
                #     dists = dists[not_outlier]

                lineWidth = None
                lineSpacing = None

                im_new = np.zeros(img_bin.shape).astype('uint8')

                starts = []
                ends = []

                distances = []
                thetas = []

                lines = []

                for _, angle, dist in zip(hspace[:], angles[:], dists[:]):
                    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
                    ax[1].plot(origin, (y0, y1), '-r')
                    distances.append(dist)
                    thetas.append(angle)
                    line = Line()
                    line.theta = np.rad2deg(angle)
                    line.distance = dist
                    line.row = y0
                    lines.append(line)

                count_beams = 0
                #check for angles in range 80 to 110, if so then a beam is present
                for angle in angles:
                    if np.rad2deg(angle) >= 80 and np.rad2deg(angle) <= 110:
                        count_beams += 1

                duration_label = "" #for all beams in the testcases
                if count_beams <= 1: duration_label = "/8"
                elif count_beams >= 2:  duration_label = "/16"

                classification_string = ""
                #now get the position with respect to the staff lines
                for i in range(len(self.quarters_bboxes)):
                    quarter_row_pos = self.quarters_bboxes[i].center[1] #get row position
                    nearest_line_idx = (np.abs(np.asarray(staff_pos) - quarter_row_pos)).argmin()
                    staff_label = staff_labels[nearest_line_idx] #a string, i.e "a1","b1",....
                    classification_string += staff_label+duration_label+" "


                #show_images([self.img_gray],[classification_string])
                return classification_string



        else: #len(self.quarters_bboxes) == 0
            #print("no quarters, symbol.row: ",self.row)
            pass



        ##### Check on half notes: ######
        if len(self.halfs_bboxes) == 1:
            #only on half note detected in this symbol
            half_row_pos = self.halfs_bboxes[0].center[1]
            nearest_line_idx = (np.abs(np.asarray(staff_pos) - half_row_pos)).argmin()
            staff_label = staff_labels[nearest_line_idx] #a string, i.e "a1","b1",....
            classification_string = staff_label #a string, i.e "a1","b1",....

            classification_string += "/2"
            #show_images([self.img_gray],[classification_string])
            return classification_string

            ################### Assuming that no chords in half notes!#######################
            # #we are certain that there is one half in the symbol
            # if 'note' in self.classification[0].lower() and 'half' not in self.classification[0].lower():
            #     #if type is note, then we are sure of our classification, just need to detect position with respect to the line
            #     classification_string += symbols_labels[self.classification[0]] #this is the ending part of the string
            # else: #our classification maybe wrong, due to the machine learning model
            #     #then we will believe the detect circles, and it will be a quarter note
            #     classification_string += "/4" #this is the ending part of the string

        elif len(self.halfs_bboxes) > 1:
            #print("more than one half notes in the same symbol")
            pass
        else: #len(self.halfs_bboxes) == 0
            #no half notes detected
            pass


        #check if the symbol is a dot using ratio of ones and zeros
        count_ones = 0
        if self.img_bin.max() <= 1:
            count_ones = np.sum(self.img_bin)
        else:
            count_ones = np.sum(self.img_bin)/255

        count_zeros = self.img_bin.shape[0]*self.img_bin.shape[1] - count_ones

        ratio = max(count_ones,count_zeros)/(self.img_bin.shape[0]*self.img_bin.shape[1])

        # a dot is detected
        if ratio > 0.8 and ratio < 1:
            classification_string = "."
            return classification_string



        #If we reached here, probably it will be an accidental, i.e "Sharp", "Flat", ..., or maybe a whole note!
        #check aspect ratio as it may be a barline, i.e check if height is way bigger than width
        if (self.img.shape[0] / self.img.shape[1]) > 5:
            #Barline
            classification_string = ""
        elif 'quarter' in self.classification[0].lower() or 'half' in self.classification[0].lower() or 'whole' in self.classification[0].lower(): #we cannot have a quarter here or a half
            nearest_line_idx = (np.abs(np.asarray(staff_pos) - self.row)).argmin()
            staff_label = staff_labels[nearest_line_idx] #a string, i.e "a1","b1",....
            classification_string = staff_label #a string, i.e "a1","b1",....

            classification_string += "/1"
        else:
            classification_string = symbols_labels[self.classification[0]]

        return classification_string
