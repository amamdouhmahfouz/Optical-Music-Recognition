from common import *

class BoundingBox:
    '''
    The resulting bounding boxes from image matching
    '''

    def __init__(self, col, row, width, height):
        self.x = col
        self.y = row
        self.width = width
        self.height = height
        self.area = width*height
        self.center = col+width/2, row+height/2 # (col,, row)
        self.img = None

    def set_img_in_bbox(self, img):
        '''
        Only sets the image (part of the whole image) that is contained inside
        this bounding box
        '''
        self.img = img
        return

    def distance(self, bbox):
        '''
        calculates the distance between 2 bounding boxes
        '''
        return np.sqrt( (self.center[0]-bbox.center[0])**2 + (self.center[1]-bbox.center[1])**2 )


    def overlapping(self, bbox):
        '''
        checks if overlapping with the other bounding box using area ratio
        '''
        overlap_width = 0
        overlap_height = 0

        max_col = 0
        if self.x > bbox.x:
            max_col = self.x
        else:
            max_col = bbox.x

        max_row = 0
        if self.y > bbox.y:
            max_row = self.y
        else:
            max_row = bbox.y

        tot_width = 0
        if self.x + self.width > bbox.x + bbox.width:
            tot_width = bbox.x + bbox.width
        else:
            tot_width = self.x + self.width

        tot_height = 0
        if self.y + self.height > bbox.y + bbox.height:
            tot_height = bbox.y + bbox.height
        else:
            tot_height = self.y + self.height

        overlap_width = tot_width - max_col
        if overlap_width < 0:
            overlap_width = 0 #to zero out the area
        overlap_height = tot_height - max_row
        if overlap_height < 0:
            overlap_height = 0 #to zero out the area

        area = overlap_width * overlap_height
        ret_area = area / self.area # compare to this ratio

        return ret_area

    def merge_boxes(self, bbox):
        '''
        merges 2 nearby boxes if possible
        '''
        x = self.x if self.x < bbox.x else bbox.x
        y = self.y if self.y < bbox.y else bbox.y
        width = self.x+self.width if self.x+self.width > bbox.x+bbox.width else bbox.x+bbox.width
        height = self.y+self.height if self.y+self.height > bbox.y+bbox.height else bbox.y+bbox.height

        ret_bbox = BoundingBox(x,y,width-x,height-y)
        return ret_bbox


    def draw_bbox(self, img):
        '''
        draws a bounding box around the given image
        '''
        cv2.rectangle(img, (int(self.x),int(self.y)), (int(self.x+self.width), int(self.y+self.height)), (255,0,0), 1)
        return
