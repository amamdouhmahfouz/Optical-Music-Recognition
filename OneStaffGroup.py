from Symbol import *
from Line import *
from common import *

class OneStaffGroup:
    '''
    A portion of the whole image that contains only 5 staff lines (i.e one staff group)
    '''
    def __init__(self):
        # everything is calculated with respect to base_row
        self.base_row = None # row coordinate of the beginning of the img from the original img
        self.start_row = None # row coordinate of the first line of the staff group
        self.end_row = None # row coordinate of the last line of the staff group
        self.symbols = [] # list of Symbols

        self.staff_space = None # the space between a line the other (adjacent ones)

        self.img_quarters = None
        self.img_gray = None
        self.img_bin = None
        self.img_beams = None
        self.img_quarters_v2 = None
        self.img_halfs = None

        self.quarters_positions = [] #sorted according to x coordinate
        self.quarters_keypoints = [] #sorted list of type Keypoints
        self.music_score = "" #the final score to be printed

        self.img_time44 = None
        self.time_signature = None

        #initiallized all by -1
        self.staff_labels_pos = {
            "c1":-1,
            "d1":-1,
            "e1":-1,
            "f1":-1,
            "g1":-1,
            "a1":-1,
            "b1":-1,
            "c2":-1,
            "d2":-1,
            "e2":-1,
            "f2":-1,
            "g2":-1,
            "a2":-1,
            "b2":-1
        }

        #using template matching
        self.matched_quarters_bboxes = None
        self.matched_halfs_bboxes = None

        #used for removing the lines correctly
        self.avg_clef_height = None

    def match_quarters(self):
        '''
        Template matching of quarter notes on this staff group
        source: opencv documentation website + stackoverflow
        '''
        img_gray = self.img_gray.copy()
        img_gray = uint8_255(img_gray)
        template = templates["quarter"][0]

        #resizing for quarters: * 1
        template_resized = resize(template, (template.shape[0] * 1, template.shape[1] * 1),
                       anti_aliasing=True)

        template_resized = uint8_255(template_resized)
        height, width = template_resized.shape[::]

        res = cv2.matchTemplate(self.img_gray, template_resized, cv2.TM_CCOEFF_NORMED)

        #for quarter, threshold = 0.55 for all images
        threshold = 0.55

        loc = np.where( res >= threshold)

        #Merge nearby bounding boxes
        symbols_locs = []
        symbols_locs.append([BoundingBox(pt[0],pt[1],width,height) for pt in zip(*loc[::-1])])
        threshold = 0.8

        bboxes = [j for i in symbols_locs for j in i]

        correct_bboxes = []
        while len(bboxes) > 0:
            r = bboxes.pop(0)
            bboxes.sort(key = lambda bb: bb.distance(r))
            merged = True
            while merged:
                merged = False
                i = 0
                for _ in range(len(bboxes)):
                    if r.overlapping(bboxes[i]) > threshold or bboxes[i].overlapping(r) > threshold:
                        r = r.merge_boxes(bboxes.pop(i))
                        merged = True
                    elif bboxes[i].distance(r) > r.width/2 + bboxes[i].width/2:
                        break
                    else: #see the rest of boxes
                        i += 1

                correct_bboxes.append(r)

        img = uint8_255(self.img_gray).copy()
        for bb in correct_bboxes:
            bb.draw_bbox(img)

        self.matched_quarters_bboxes = []
        #remove duplicates if any from correct_bboxes
        [self.matched_quarters_bboxes.append(x) for x in correct_bboxes if x not in self.matched_quarters_bboxes]

        # fit the bounding boxes to their respective symbols
        for symbol in self.symbols:
            symbol.fit_quarters_bounding_boxes(self.matched_quarters_bboxes)


        return

    def match_halfs(self):
        '''
        Template matching of half notes on this staff group
        source: opencv documentation website
        '''
        img_gray = self.img_gray.copy()
        img_gray = uint8_255(img_gray)
        template = templates["half"][0]

        #resizing for halfs: * 1
        template_resized = resize(template, (template.shape[0] * 1, template.shape[1] * 1),
                       anti_aliasing=True)

        template_resized = uint8_255(template_resized)
        height, width = template_resized.shape[::]

        res = cv2.matchTemplate(self.img_gray, template_resized, cv2.TM_CCOEFF_NORMED)

        #for quarter, threshold = 0.55 for all images
        threshold = 0.55 #For TM_CCOEFF_NORMED, larger values = good fit.

        loc = np.where( res >= threshold)


        #Merge nearby bounding boxes into one box
        symbols_locs = []
        symbols_locs.append([BoundingBox(pt[0],pt[1],width,height) for pt in zip(*loc[::-1])])
        threshold = 0.8

        bboxes = [j for i in symbols_locs for j in i]

        correct_bboxes = []
        while len(bboxes) > 0:
            r = bboxes.pop(0)
            bboxes.sort(key = lambda bb: bb.distance(r))
            merged = True
            while merged:
                merged = False
                i = 0
                for _ in range(len(bboxes)):
                    if r.overlapping(bboxes[i]) > threshold or bboxes[i].overlapping(r) > threshold:
                        r = r.merge_boxes(bboxes.pop(i))
                        merged = True
                    elif bboxes[i].distance(r) > r.width/2 + bboxes[i].width/2:
                        break
                    else: #see the rest of boxes
                        i += 1

                correct_bboxes.append(r)

        img = uint8_255(self.img_gray).copy()
        for bb in correct_bboxes:
            bb.draw_bbox(img)

        self.matched_halfs_bboxes = []
        #remove duplicates if any from correct_bboxes
        [self.matched_halfs_bboxes.append(x) for x in correct_bboxes if x not in self.matched_halfs_bboxes]

        # fit the bounding boxes to their respective symbols
        for symbol in self.symbols:
            symbol.fit_halfs_bounding_boxes(self.matched_halfs_bboxes)

        return

    def classify_symbols(self, model):
        '''
        classify the type of the symbol, i.e 'G-Clef', 'Quarter-Note', ... etc
        '''
        # loop on all symbols of this staff group
        for symbol in self.symbols:
            symbol.classify_symbol(model)

        return

    def show(self):
        plt.figure(figsize=(15,20))
        io.imshow(self.img_gray)
        return



    def assign_labels_to_lines(self):
        '''
        sets the member variable: self.staff_labels_pos
        '''

        base = self.base_row
        start_line_row = self.start_row
        end_line_row = self.end_row

        #make coordinates with respect to current staff group image not the whole image
        start_line_row = start_line_row - base
        end_line_row = end_line_row - base

        #we can now easily get the rest of line positions using staff_space
        staff_space = self.staff_space

        self.staff_labels_pos["b2"] = int(start_line_row - staff_space - staff_space/2)
        self.staff_labels_pos["a2"] = start_line_row - staff_space
        self.staff_labels_pos["g2"] = int(start_line_row - staff_space/2)
        self.staff_labels_pos["f2"] = start_line_row
        self.staff_labels_pos["e2"] = int(start_line_row + 1*staff_space/2)
        self.staff_labels_pos["d2"] = int(start_line_row + 2*staff_space/2)
        self.staff_labels_pos["c2"] = int(start_line_row + 3*staff_space/2)
        self.staff_labels_pos["b1"] = int(start_line_row + 4*staff_space/2)
        self.staff_labels_pos["a1"] = int(start_line_row + 5*staff_space/2)
        self.staff_labels_pos["g1"] = int(start_line_row + 6*staff_space/2)
        self.staff_labels_pos["f1"] = int(start_line_row + 7*staff_space/2)
        self.staff_labels_pos["e1"] = int(start_line_row + 8*staff_space/2)
        self.staff_labels_pos["d1"] = int(start_line_row + 9*staff_space/2)
        self.staff_labels_pos["c1"] = int(start_line_row + 10*staff_space/2)

        return


    def set_time_signature(self):
        '''
        sets the time signature of this staff group if any
        '''
        for symbol in self.symbols:
            if "time" in symbol.classification[0].lower():
                self.time_signature = symbol.classification
        return


    def get_scores(self):
        '''
        gets the musical score in the this staff group from all symbols that it has
        Note: symbols are already sorted in ascending order by columns
        '''
        music_score = "["
        for symbol in self.symbols:
            music_score += symbol.score(self.staff_labels_pos)+" "

        music_score += "]"
        return music_score




    def detect_quarters(self):
        '''
        detect quarter notes in this staff_group's image
        source: openv documentation website
        This method is not used in our pipeline, but is left for future reference if
        we would like to see the effect of different methods
        '''

        morph_size = 7 #should depend on the size of the image
        morph_elem = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(morph_elem, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
        operation = cv2.MORPH_CLOSE
        self.img_quarters_v2 = cv2.morphologyEx(self.img_gray, operation, element)

        # Read image
        im = self.img_quarters_v2.copy()#dst.copy()

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 3

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.0001 #first set it with a very small number, then of there is cirlce that is relatively large to other circles, apply dilation, then increase convexity to 0.5 dot detect good ellipses

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else :
            detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(im)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob
        self.img_quarters_v2 = cv2.drawKeypoints(im, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #keypoints have the following attributes:
        #  - size: size of the circle
        #  - pt: coordinates of the circle in the form of (x,y) not (y,x), i.e not the conventional
        # first sort the keypoins according to its column position in increasing order
        keypoints.sort(key = lambda point: point.pt[0], reverse=False) #sorts in-place
        self.quarters_positions += [point.pt[::-1] for point in keypoints]
        self.quarters_keypoints = keypoints.copy()

        #fit the quarters_positions and quarters_keypoints in each symbol
        for symbol in self.symbols:
            symbol.fit_quarters_and_keypoints(self.quarters_positions,self.quarters_keypoints)


        return


    # TODO: correct this if needed
    def __str__(self):
        print("OneStaffGroup:: start_row=",self.start_row,", end_row=",self.end_row)
