from common import *
from OneStaffGroup import *


class SheetMusic:
    '''
    Contains the whole music sheet.
    Has a list of OneStaffGroups.
    '''
    def __init__(self, img_path, model_path='nn86_model.sav'):
        self.avg_theta = None # angle in degrees
        self.num_staff_groups = 0 # number of staff lines groups
        self.staff_space = None # the space between each line and the other belonging to the same staff group
        self.staff_height = None # thickness of one line
        self.lines = [] # list of Lines, which have theta and distance
        self.start_row = None # first line y coordinate of ALL the lines
        self.end_row = None # last line y coordinate of ALL the lines
        self.staff_groups = [] # list of OneStaffGroup


        self.lines_positions = [] # row position of each line
        self.barriers = [] # list of y coordinates, represents a line the divide between staff_groups

        self.img_path = img_path
        self.img_rgb = io.imread(self.img_path)
        self.img_gray = (rgb2gray(self.img_rgb)*255).astype('uint8')
        thres = threshold_yen(self.img_gray)
        self.img_bin = ~(self.img_gray > thres)


        self.img_segmented = None
        self.staff_groups_centroids = []
        self.model = pickle.load(open(model_path, 'rb')) #'nn825_model.sav'
        self.target_img_size = (32, 32)
        self.time_signatures = []




    def compute_avg_theta(self):
        '''
        Gets average rotation angle, number of staff groups, valid staff lines only
        '''
        angles = []
        distances = []
        assert len(self.lines) > 0, "ERROR::NO_LINES_ADDED_TO_STAFF"
        for line in self.lines:
            angles.append(line.theta)
            distances.append(line.distance)

        #histogram of 36 bins, i.e angles accuracy is +-10 degrees
        angle_resolution = 5 #accuracy of angles
        n_bins = 360 // angle_resolution
        hist = np.zeros(n_bins)
        hist_neg = np.zeros(n_bins) #histogram of negative angles
        for angle in angles:
            if angle >= 0:
                hist[abs(int(angle//angle_resolution))] += 1
            else:
                hist_neg[abs(int(angle//angle_resolution))] += 1


        if np.max(hist) > np.max(hist_neg):
            #get the peak of the histogram, this will correspond to the avg_theta
            self.avg_theta = np.argmax(hist) * angle_resolution #returns the index of max, index=theta
        else:
            self.avg_theta = -np.argmax(hist_neg) * angle_resolution

        correct_lines_positions = []
        correct_distances = []
        for i in range(len(distances)):
            #check if the corresponding angle falls in the avg_theta bin
            if angles[i] >= self.avg_theta and angles[i] < self.avg_theta + angle_resolution:
                correct_distances.append(distances[i])
                correct_lines_positions.append(self.lines[i].row)

        self.num_staff_groups = len(correct_distances) // 5
        self.lines_positions = correct_lines_positions

        correct_distances = np.sort(correct_distances)[::-1] #sort descendingly
        #assert len(correct_distances) >= 5, "ERROR::TOO_FEW_STAFF_LINES i.e less than 5 staff lines in the image"


        #assuming that maximum difference is 40
        max_diff = 40
        diff_resolution = 6
        avg_distances = np.zeros(max_diff * diff_resolution)
        diff_hist = np.zeros(max_diff * diff_resolution) #histogram of differences of adjacent elements
        avg_dict = {}
        for i in range(len(correct_distances)-1):
            bin_diff = (int(abs(correct_distances[i+1] - correct_distances[i])) // diff_resolution)
            diff = abs(correct_distances[i+1] - correct_distances[i])
            diff_hist[bin_diff] += 1
            if bin_diff not in avg_dict.keys():
                avg_dict[bin_diff] = []
                avg_dict[bin_diff].append(diff)
            else:
                avg_dict[bin_diff].append(diff)

        self.staff_space = int(np.mean(avg_dict[np.argmax(diff_hist)]))

        return self.avg_theta



    def assign_staff_groups(self):
        '''
        Now we have all lines
        The function filters these lines
        '''


        sorted_lines_positions = np.sort(self.lines_positions)#[::-1]
        self.start_row = sorted_lines_positions[0]
        self.end_row = sorted_lines_positions[-1]
        #we have row position of each line, num_staff_groups

        for i in range(self.num_staff_groups):
            staff_group = OneStaffGroup()
            staff_group.start_row = sorted_lines_positions[i*5]
            staff_group.end_row = sorted_lines_positions[i*5+4]
            staff_group.staff_space = self.staff_space
            self.staff_groups.append(staff_group)
            self.staff_groups_centroids.append( (staff_group.start_row+staff_group.end_row)/2 )

        self.staff_groups_centroids = np.sort(self.staff_groups_centroids)

        # compute the barriers
        if self.num_staff_groups == 1:
            #print("Only one staff group found")
            self.barriers.append(self.img_bin.shape[0])
        else:
            #print("Found ", self.num_staff_groups, " staff groups")
            for i in range(len(self.staff_groups)-1):
                y1 = self.staff_groups[i].end_row
                y2 = self.staff_groups[i+1].start_row
                y_mid = int( (y1+y2)/2 )
                self.barriers.append( y_mid )


        #assign to each staff group its part of the image
        #for staff_group in self.staff_groups:
        for i in range(self.num_staff_groups):
            if i == 0:
                self.staff_groups[i].base_row = 0
                self.staff_groups[i].img_gray = self.img_gray[0:self.barriers[0],:]
                self.staff_groups[i].img_bin = self.img_bin[0:self.barriers[0],:]
                self.staff_groups[i].assign_labels_to_lines()
            elif i == self.num_staff_groups-1:
                self.staff_groups[i].base_row = self.barriers[-1]
                self.staff_groups[i].img_gray = self.img_gray[self.barriers[-1]:,:]
                self.staff_groups[i].img_bin = self.img_bin[self.barriers[-1]:,:]
                self.staff_groups[i].assign_labels_to_lines()
            else:
                self.staff_groups[i].base_row = self.barriers[i-1]
                self.staff_groups[i].img_gray = self.img_gray[self.barriers[i-1]:self.barriers[i],:]
                self.staff_groups[i].img_bin = self.img_bin[self.barriers[i-1]:self.barriers[i],:]
                self.staff_groups[i].assign_labels_to_lines()

        # Now we have all staff groups with their starting and ending row positions

        #print(self.num_staff_groups," staff groups are detected")

        return


    def rotate(self):
        '''
        rotates the whole music sheet
        '''

        # Rotate image with angle 90-theta
        avg_angle = self.avg_theta
        if avg_angle <0: avg_angle = 360 - avg_angle
        rotation_angle = 90-avg_angle

        self.img_bin = imutils.rotate_bound(uint8_255(self.img_bin*1), -rotation_angle)
        self.img_rgb = uint8_255(imutils.rotate_bound(self.img_rgb, -rotation_angle))
        self.img_gray = uint8_255(imutils.rotate_bound(img_as_ubyte(self.img_gray), -rotation_angle))

        return


    def show(self, mode='rgb'):
        plt.figure(figsize=(15,20))
        if mode == "rgb":
            io.imshow(self.img_rgb)
        elif mode == "gray":
            io.imshow(self.img_gray)
        elif mode == "binary":
            io.imshow(self.img_bin, cmap="gray")
        elif mode == "segmented":
            io.imshow(self.img_segmented)
        else:
            io.imshow(self.img_bin, cmap="gray")

        return

    def add_symbol(self, img_symbol, x1, y1, x2, y2, img_bin, img_gray):
        '''
        Adds the symbol to its belonging staff group
        inputs:
            x1: top left corner col
            y1: top left corner row
            x2: bottom right corner col
            y2: bottom right corner row
            NOTE: x1,y1,x2,y2 are all coordinates with respect to the whole image not the staff group
            self.base_row of each staff group is now set
        '''

        if self.num_staff_groups == 1: #only one staff group
            symbol = Symbol(img_symbol, x1, y1, x2, y2, img_bin, img_gray)
            symbol.base_row = 0
            self.staff_groups[0].symbols.append(symbol)
            return

        symbol = Symbol(img_symbol, x1, y1, x2, y2, img_bin, img_gray)
        # centroids of the symbol are symbol.row and symbol.col, with respect to the whole image
        # we are only interested in the y coordinate, i.e symbol.row
        # get the nearest group index to this symbol
        nearest_group_idx = (np.abs(np.asarray(self.staff_groups_centroids) - symbol.row)).argmin()
        #print("Symbol belongs to staff group: ", nearest_group_idx)
        #print("Symbol.row: ", symbol.row)
        #print("Symbol cols: x1=",symbol.x1, ", x2=",symbol.x2)
        symbol.base_row = self.staff_groups[nearest_group_idx].base_row
        self.staff_groups[nearest_group_idx].symbols.append(symbol)

        return



    def segment(self):
        '''
        Osama's approach of segmentation :)
        '''
        #from skimage.color import rgba2rgb
        def Binary(Image, Threshold):
            GrayCoins = rgb2gray(Image)
            BinaryCoins = GrayCoins.copy()
            Height, Width = Image.shape[0:2]
            BinaryCoins = (GrayCoins <= Threshold)
            return BinaryCoins

        image = self.img_rgb.copy() #cv2.imread(self.img_path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        thresh = self.img_bin#cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove horizontal
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image, [c], -1, (255,255,255), 2)

        # Repair image
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,8))
        result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=2)

        result = Binary(result, 0.5)
        self.img_segmented = result.copy()
        #plt.figure(figsize=(10,15))
        #show_images([image, result], ['Image', 'Result'])
        return


    def detect_lines(self):
        '''
        gets all valid lines using hough transform
        '''


        # Detecting staff lines to get the angle of rotation
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
        h, theta, d = hough_line(self.img_bin, theta=tested_angles)

        origin = np.array((0, self.img_bin.shape[1]))
        hspace, angles, dists = hough_line_peaks(h, theta, d)

        ###### Remove outliers ########
        ## can remove using histograms, where we see the most angles we have
        # i.e removing false lines
#         mean = np.mean(angles)
#         standard_deviation = np.std(angles)
#         distance_from_mean = abs(angles - mean)
#         max_deviations = 2
#         not_outlier = distance_from_mean < max_deviations * standard_deviation
#         if standard_deviation > 0.08: #remove outliers
#             angles = angles[not_outlier]
#             hspace = hspace[not_outlier]
#             dists = dists[not_outlier]

        lineWidth = None
        lineSpacing = None

        lines = []
        for _, angle, dist in zip(hspace[:], angles[:], dists[:]):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            #ax[1].plot(origin, (y0, y1), '-r')
            line = Line()
            line.theta = np.rad2deg(angle)
            line.distance = dist
            line.row = y0
            lines.append(line)

        self.lines = lines.copy()

        return


    def detect_symbols(self):
        # remove artifacts connected to image border
        cleared = self.img_segmented.copy()#clear_border(bw)
        result = self.img_segmented.copy()

        # label image regions
        label_image = label(cleared) #connected components algorithm
        #show_images([label_image])
        # to make the background transparent, pass the value of `bg_label`,
        # and leave `bg_color` as `None` and `kind` as `overlay`
        image_label_overlay = label2rgb(label_image, image=result, bg_label=0) #convert to 3 dimensional colored image

        #fig, ax = plt.subplots(figsize=(10, 6))
        #ax.imshow(image_label_overlay)
        rects = []
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 10:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                #print(region.bbox)

                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                rects.append(rect)

        symbols = [] #will store symbols temporarily

        a = 0
        rows = len(rects)
        cols = 1
        axes = []
        #plt.figure(figsize=(15,13))
        #add each symbol to its corresponding staff group
        for i in range(len(rects)):
            box_num = i
            img_symbol = result[rects[box_num].xy[1] : rects[box_num].xy[1]+rects[box_num].get_height(),
                               rects[box_num].xy[0] : rects[box_num].xy[0]+rects[box_num].get_width()]
            img_bin_symbol = self.img_bin[rects[box_num].xy[1] : rects[box_num].xy[1]+rects[box_num].get_height(),
                               rects[box_num].xy[0] : rects[box_num].xy[0]+rects[box_num].get_width()]

            img_gray_symbol = self.img_gray[rects[box_num].xy[1] : rects[box_num].xy[1]+rects[box_num].get_height(),
                               rects[box_num].xy[0] : rects[box_num].xy[0]+rects[box_num].get_width()]

            _symbol = Symbol(img_symbol, rects[box_num].xy[0], rects[box_num].xy[1],
                        rects[box_num].xy[0]+rects[box_num].get_width(), rects[box_num].xy[1]+rects[box_num].get_height(),
                         img_bin_symbol, img_gray_symbol)
            symbols.append(_symbol)

        #classify symbols list to get the g-clefs
        for _symbol in symbols:
            _symbol.classify_symbol(self.model)
            #now for each _symbol in symbols list, they have their classifications

        #find g-clefs in from symbols list
        num_g_clefs = 0
        col_pos = []
        for _symbol in symbols:
            if _symbol.classification == 'G-Clef':
                #print("Found a G-Clef at col: ",_symbol.col)
                num_g_clefs += 1
                col_pos.append(_symbol.col)


        H = np.zeros(int(self.img_bin.shape[1]/5))
        resolution = 5
        for c in col_pos:
            H[int(c/resolution)] += 1
        max_idx = np.argmax(H)
        comparison_num = resolution * max_idx #to get the approximate number

        for _symbol in symbols:
            if _symbol.classification == 'G-Clef':
                if (_symbol.col >= comparison_num and
                _symbol.col < comparison_num + resolution):
                    #print("G-Clef at col: ",_symbol.col)
                    pass
                else: #misclassified as 'G-Clef'
                    _symbol.classification = ['Unknown']
                    _symbol.doubt = True

        #now compute the average height of g-clefs
        avg_clef_height = 0
        count_g_clefs = 0
        for _symbol in symbols:
            if _symbol.classification[0] == "G-Clef":
                count_g_clefs += 1
                height = _symbol.y2 - _symbol.y1
                avg_clef_height += height

        avg_clef_height /= len(self.staff_groups)
        self.avg_clef_height = avg_clef_height

        return



    def resegment(self):
        '''
        Resegment using a better approach depending the height of the G-Clef
        '''
        def Binary(Image, Threshold):
            GrayCoins = rgb2gray(Image)
            BinaryCoins = GrayCoins.copy()
            BinaryCoins = (GrayCoins <= Threshold)
            return BinaryCoins

        image = cv2.imread(self.img_path) #Put the Original Input Image Here
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove horizontal
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (250,1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image, [c], -1, (255,255,255), 4)

        #Min = 129, Max = 400
        # Repair image
        #print(self.avg_clef_height)
        CliffSize = self.avg_clef_height #Cliff size here 😃
        if self.avg_clef_height is None or self.avg_clef_height == 0:
            CliffSize = self.img_bin.shape[0]-40 #280 in all cases
        #print("CliffSize: ",CliffSize)
        Tuner = round(CliffSize/100)
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6+Tuner))
        result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

        result = Binary(result, 0.5)
        self.img_segmented = result.copy()

        #show_images([image, result], ['Image', 'Result of Resegmentation'])
        return

    def redetect_symbols(self):
        '''
        Redetect symbols based upon the resegmentation, i.e a better approach :)
        '''
        # remove artifacts connected to image border
        cleared = self.img_segmented.copy()#clear_border(bw)
        result = self.img_segmented.copy()

        # label image regions
        label_image = label(cleared) #connected components algorithm
        #show_images([label_image])
        # to make the background transparent, pass the value of `bg_label`,
        # and leave `bg_color` as `None` and `kind` as `overlay`
        image_label_overlay = label2rgb(label_image, image=result, bg_label=0) #convert to 3 dimensional colored image

        rects = []
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 60:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                rects.append(rect)

        a = 0
        rows = len(rects)
        cols = 1
        axes = []
        #plt.figure(figsize=(15,13))
        #add each symbol to its corresponding staff group
        for i in range(len(rects)):
            box_num = i
            img_symbol = result[rects[box_num].xy[1] : rects[box_num].xy[1]+rects[box_num].get_height(),
                               rects[box_num].xy[0] : rects[box_num].xy[0]+rects[box_num].get_width()]
            img_bin_symbol = self.img_bin[rects[box_num].xy[1] : rects[box_num].xy[1]+rects[box_num].get_height(),
                               rects[box_num].xy[0] : rects[box_num].xy[0]+rects[box_num].get_width()]

            img_gray_symbol = self.img_gray[rects[box_num].xy[1] : rects[box_num].xy[1]+rects[box_num].get_height(),
                               rects[box_num].xy[0] : rects[box_num].xy[0]+rects[box_num].get_width()]
            self.add_symbol(img_symbol, rects[box_num].xy[0], rects[box_num].xy[1],
                       rects[box_num].xy[0]+rects[box_num].get_width(),
                       rects[box_num].xy[1]+rects[box_num].get_height(), img_bin_symbol,img_gray_symbol)

        return


    def reclassify_symbols(self):
        '''
        Reclassify symbols based on the new segmentation
        '''
        for staff_group in self.staff_groups:
            staff_group.classify_symbols(self.model)
        return

    # extract features from these symbols
    def extract_hog_features(self):
        """
        Extracts Histogram Of Features from input image
        """
        img = cv2.resize(img, target_img_size)
        win_size = (32, 32)
        cell_size = (4, 4)
        block_size_in_cells = (2, 2)

        block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
        block_stride = (cell_size[1], cell_size[0])
        nbins = 9  # Number of orientation bins
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        h = hog.compute(img)
        h = h.flatten()
        return h.flatten()

    def classify_symbols(self):
        '''
        Classify the type of each symbol, i.e 'Quarter-Note', 'G-Clef',...etc
        '''

        for staff_group in self.staff_groups:
            staff_group.classify_symbols(self.model)

        return

    def show_classifications(self):
        for staff_group in self.staff_groups:
            for symbol in staff_group.symbols:
                print(symbol.classification)
        return


    def find_g_clefs(self):
        '''
        Finds the CORRECT G-Clefs in the music sheet,
        Assuming the each staff group must start with a G-Clef
        '''
        num_g_clefs = 0
        col_pos = []
        for staff_group in self.staff_groups:
            for symbol in staff_group.symbols:
                if symbol.classification == 'G-Clef':
                    #print("Found a G-Clef at col: ",symbol.col)
                    num_g_clefs += 1
                    col_pos.append(symbol.col)


        H = np.zeros(int(self.img_bin.shape[1]/5))
        resolution = 5
        for c in col_pos:
            H[int(c/resolution)] += 1
        max_idx = np.argmax(H)
        comparison_num = resolution * max_idx #to get the approximate number

        for staff_group in self.staff_groups:
            for symbol in staff_group.symbols:
                if symbol.classification == 'G-Clef':
                    if (symbol.col >= comparison_num and
                    symbol.col < comparison_num + resolution):
                        pass
                        #print("G-Clef at col: ",symbol.col)
                    else: #misclassified as 'G-Clef'
                        symbol.classification = ['Unknown']
                        symbol.doubt = True

        return


    def sort_symbols_in_staffs(self):
        '''
        Sorts the symbols in each staff group in increasing order by their col position
        '''

        for i in range(len(self.staff_groups)):
            #sort in-place the symbols ascendingly according to the col position
            self.staff_groups[i].symbols.sort(key = lambda symbol: symbol.col, reverse=False)

        return

    def show_sorted_symbols(self):
        c = 0
        for staff_group in self.staff_groups:
            print("Staff Group: #",c)
            c+=1
            for symbol in staff_group.symbols:
                print(symbol.col)

        return

    def set_time_signature(self):
        '''
        checks if there is a time signature in the sheet or not
        '''
        for staff_group in self.staff_groups:
            staff_group.set_time_signature()

        for staff_group in self.staff_groups:
            for symbol in staff_group.symbols:
                if "time" in symbol.classification[0].lower():
                    self.time_signatures.append(symbol.classification)

        return


    def detect_quarters(self):
        '''
        detect all full note heads, i.e quarter notes
        using morphology
        '''

        for staff_group in self.staff_groups:
            staff_group.detect_quarters()

        return

    def match_quarters(self):
        '''
        Template matching of quarter notes on all staff groups in the SheetMusic
        source: opencv documentation website
        '''
        for staff_group in self.staff_groups:
            staff_group.match_quarters()

        return

    def match_halfs(self):
        '''
        Template matching of half notes on all staff groups in the SheetMusic
        source: opencv documentation website
        '''
        for staff_group in self.staff_groups:
            staff_group.match_halfs()

        return


    def get_avg_clef_height(self):
        '''
        sets the average clef height if we have more than one staff_group
        Used for resegmentation if staff lines were not detected correctly
        '''
        avg_clef_height = 0
        for staff_group in self.staff_groups:
            for symbol in staff_group.symbols:
                if symbol.classification[0] == "G-Clef":
                    height = symbol.y2 - symbol.y1
                    avg_clef_height += height

        avg_clef_height /= len(self.staff_groups)
        self.avg_clef_height = avg_clef_height

        return

    def get_scores(self):
        '''
        return the classification score of the music sheet
        '''

        starter = ""
        ender = ""
        score = ""
        if len(self.staff_groups) > 1:
            starter = "{"
            ender = "}"
        score += starter
        for staff_group in self.staff_groups:
            score += staff_group.get_scores() + ","

        score = score[:-1] + ender
        return score

    def run(self):
        '''
        Runs the whole system and return an output string
        '''
        self.detect_lines()
        self.compute_avg_theta()
        self.rotate()
        self.assign_staff_groups()
        self.segment()
        self.detect_symbols()
        self.resegment()
        self.redetect_symbols()
        self.reclassify_symbols()
        self.find_g_clefs()
        self.sort_symbols_in_staffs()
        self.set_time_signature()
        self.match_quarters()
        self.match_halfs()
        music_score = self.get_scores()
        music_score = format_string(music_score)


        return music_score



def format_chords(str1):
    TestString = str1
    NewString = ''
    NumberOfCharacters = 0
    STRLEN = len(TestString)
    RemovedBraces = 0
    if TestString[0] == '{':
        RemovedBraces = 1
        NewString = TestString[1:STRLEN-1]
    else:
        NewString = TestString

    FoundBracket = 0
    DummyString = []
    FinalString = ''
    TempString = ''
    for A in NewString:
        if A == '{':
            FinalString = FinalString + A
            FoundBracket = 1
        elif A == '}':
            #Sorting Here
            DummyString.append(TempString)
            TempString = ''
            while len(DummyString) > 0:
                Min = 1000
                MinIndex = 0
                for i in range(len(DummyString)):
                    if ord(DummyString[i][0]) < Min:
                        Min = ord(DummyString[i][0])
                        MinIndex = i

                FinalString = FinalString + DummyString[MinIndex]
                if len(DummyString) > 1:
                    FinalString = FinalString + ','
                del DummyString[MinIndex]

            FinalString = FinalString + A
            FoundBracket = 0
        else:
            if FoundBracket == 1:
                if A == ',':
                    TempString.replace(' ', '')
                    DummyString.append(TempString)
                    TempString = ''
                else:
                    TempString = TempString + A
            else:
                FinalString = FinalString + A

    if RemovedBraces == 1:
        FinalString = '{' + FinalString
        FinalString = FinalString + '}'

    Output = FinalString

    return Output


def format_string(str1):
    TestString = str1
    NewString = ''
    NumberOfCharacters = 0
    for A in TestString:
        if ord(A) > 96 and ord(A) < 123:
            NewString = NewString + A
            for j in range(NumberOfCharacters):
                NewString = NewString + '#'
            NumberOfCharacters = 0
        elif A == '#':
            NumberOfCharacters = NumberOfCharacters + 1
        else:
            NewString = NewString + A

    NewString2 = ''
    NumberOfCharacters = 0
    for A in NewString:
        if ord(A) > 96 and ord(A) < 123:
            NewString2 = NewString2 + A
            for j in range(NumberOfCharacters):
                NewString2 = NewString2 + '&'
            NumberOfCharacters = 0
        elif A == '&':
            NumberOfCharacters = NumberOfCharacters + 1
        else:
            NewString2 = NewString2 + A

    chords_sorted = format_chords(NewString2)

    return chords_sorted


if __name__ == "__main__":
    img_path = sys.argv[1]
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    sheet = SheetMusic(img_path)
    music_score = sheet.run()

    print(music_score)
