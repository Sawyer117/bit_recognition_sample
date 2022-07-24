# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 16:13:33 2022

@author: 36284
"""

import cv2


def template_matching(img,template):
    #img2 = img.copy()
    #template = cv2.imread(template_dir,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #print("line 120: {}".format(template.shape))
    w, h = template.shape[::-1]
    
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'] #'cv.TM_CCORR' this is good
    methods = ['cv2.TM_SQDIFF_NORMED']
    for meth in methods:
        #start_time = time.time()
        #img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        x,y = top_left[0], top_left[1]
        w,h = bottom_right[0]-x, bottom_right[1]-y
        return top_left, bottom_right, [w, h]

#def draw_rect(img=None,star):
#    
#    #image = cv2.rectangle(img, start_point, end_point, color, thickness)
#    return True
    
if __name__ == "__main__":
    img_name =  "./images/3.jpg"
    template_name = "./images/template_3.jpg"
    img = cv2.imread(img_name)
    template = cv2.imread(template_name)
    top_left, bottom_right, [w,h] = template_matching(img, template)
    color = (255, 0, 0)
    thickness = 3
    rect_image = cv2.rectangle(img, top_left, bottom_right, color, thickness)
    dsize = (int(0.3*img.shape[1]), int(0.3*img.shape[0]))
    # resize image
    rect_image = cv2.resize(rect_image, dsize)
    cv2.imshow('image window', rect_image)
    # add wait key. window waits until user presses a key
    cv2.waitKey(0)
    # and finally destroy/close all open windows
    cv2.destroyAllWindows()