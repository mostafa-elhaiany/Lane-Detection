import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os



#using plt to show frames but colors come out inverted when reading with cv2 due to it reading in bgr format
def display(images): 
    plt.figure(figsize=(40,40))
    for i, image in enumerate(images):
        plt.subplot(3,2,i+1)
        plt.imshow(image, None)
        plt.autoscale(tight=True)
    plt.show()

def display2(images): #showing a list of numpy frames
    i=1
    for image in images:
        cv2.imshow('image '+str(i),image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        i+=1
    
def getImages(file): #getting squences of images in a numpy array
    images= os.listdir(file)
    imglist=[]
    for image in images:
        img = cv2.imread(file+image)
        imglist.append(img)
    return imglist

def preprocess1(image):
    hlsImage = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([0,190,0])
    upper = np.array([255,255,255])
    yellowlower = np.array([10,0,90])
    yellowupper = np.array([50,255,255])
    yellowmask = cv2.inRange(hlsImage, yellowlower, yellowupper)
    whitemask = cv2.inRange(hlsImage, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def Filter(images): #map function for the hls filtering function
    newImages=[]
    for image in images:
        newImage = preprocess1(image)
        newImages.append(newImage)
    return newImages    
    
def getVertices(img):
    x,y = img.shape[1],img.shape[0]
    shape = np.array([[0, y], [x, y], [int(0.55*x), int(0.6*y)], [int(0.45*x), int(0.6*y)]])
    return np.int32([shape])

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def preprocess2(images): #map function for the roi image function
    newImages=[]
    for image in images:
        vertices = getVertices(image)
        newImage= region_of_interest(image,vertices)
        newImages.append(newImage)
    return newImages 


   #Applies the Canny transform
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#map function for the Canny edge detection function
def preprocess3(images): 
    grayImages=[]
    for image in images:
        grayImage= grayscale(image)
        grayImages.append(grayImage)
    newImages=[]
    for image in grayImages:
        newImage= canny(image,50,100)
        newImages.append(newImage)
    return newImages 



rightSlope, leftSlope, rightIntercept, leftIntercept = [],[],[],[]
def draw_lines(img, lines, thickness=5):
    global rightSlope, leftSlope, rightIntercept, leftIntercept
    rightColor=[255,0,0]   
    leftColor=[255,0,0]
    
    #this is used to filter out the outlying lines that can affect the average
    #We then use the slope we determined to find the y-intercept of the filtered lines by solving for b in y=mx+b
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y1-y2)/(x1-x2)
            if slope > 0.3:
                if x1 > 500 :
                    yintercept = y2 - (slope*x2)
                    rightSlope.append(slope)
                    rightIntercept.append(yintercept)
                else: None
            elif slope < -0.3:
                if x1 < 600:
                    yintercept = y2 - (slope*x2)
                    leftSlope.append(slope)
                    leftIntercept.append(yintercept)
    #We use slicing operators and np.mean() to find the averages of the 30 previous frames
    #This makes the lines more stable, and less likely to shift rapidly
    leftavgSlope = np.mean(leftSlope[-30:])
    leftavgIntercept = np.mean(leftIntercept[-30:])
    rightavgSlope = np.mean(rightSlope[-30:])
    rightavgIntercept = np.mean(rightIntercept[-30:])
    #Here we plot the lines and the shape of the lane using the average slope and intercepts
    try:
        left_line_x1 = int((0.65*img.shape[0] - leftavgIntercept)/leftavgSlope)
        left_line_x2 = int((img.shape[0] - leftavgIntercept)/leftavgSlope)
        right_line_x1 = int((0.65*img.shape[0] - rightavgIntercept)/rightavgSlope)
        right_line_x2 = int((img.shape[0] - rightavgIntercept)/rightavgSlope)
        pts = np.array([[left_line_x1, int(0.65*img.shape[0])],[left_line_x2, int(img.shape[0])],[right_line_x2, int(img.shape[0])],[right_line_x1, int(0.65*img.shape[0])]], np.int32)
        pts = pts.reshape((-1,1,2))
        
        cv2.line(img, (left_line_x1, int(0.65*img.shape[0])), (left_line_x2, int(img.shape[0])), leftColor, 10)
        cv2.line(img, (right_line_x1, int(0.65*img.shape[0])), (right_line_x2, int(img.shape[0])), rightColor, 10)
    except ValueError:
    #I keep getting errors for some reason, so I put this here. Idk if the error still persists.
        pass
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

#######################
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def linedetect(img):
    return hough_lines(img, 1, np.pi/180, 20, 20, 300)

def getHough(images): #map function for the lineDetect function
    newImages=[]
    for image in images:
        img=linedetect(image)
        newImages.append(img)
    return newImages
    


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):

    return cv2.addWeighted(initial_img, α, img, β, γ)

def finalOut(lines,initial): #map function for the weighted image function
    newImages=[]
    i=0
    n=len(lines)
    while(i<n):
        weighted=weighted_img(lines[i],initial[i])
        newImages.append(weighted)
        i+=1
    return newImages



def laneDetection(file): #given a directory with an image sequence outputs images with the new weighted images.
    imageList=getImages(file)
    processedImages= Filter(imageList)
    roiImages=preprocess2(processedImages)
    cannyImages=preprocess3(roiImages)
    
    hough_img = getHough(cannyImages)
    final=finalOut(hough_img,imageList)

    return final
 


def laneDetect(image):
    processed=preprocess1(image)
    vertices=getVertices(processed)
    roi=region_of_interest(processed,vertices)
    gray= grayscale(roi)
    Cann=canny(gray,50,100)
    line=linedetect(Cann)
    final=weighted_img(line,image)
    return final


def vid(file):
    cap = cv2.VideoCapture(file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i=0
    frames=[]
    #frame_width = int(cap.get(3))  # uncomment if we wanna save a video
    #frame_height = int(cap.get(4))
   # videoOut = cv2.VideoWriter('output4.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

    while(cap.isOpened() and i<length-2):
        ret, frame = cap.read()
         
        if(frame is not None):
            out=laneDetect(frame)
           # videoOut.write(out)
            cv2.imshow('out',out)
            
        
        i+=1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    return frames
    
vid1=vid('1.mp4') #video name or directory here
