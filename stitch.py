import cv, math, sys, numpy, time
from progressbar import ProgressBar

NO_NEIGHBOR = -1

#convert this C code to python

def rectContains(rect, point):
    # rect is tuple of width and height
    # point is an x, y point
    if ((point[0]<0) or (point[1]<0)):
        return False
    if ((point[0]<rect[0]) and (point[1]<rect[1])):
        return True
    else:
        return False

def applyHomographyToPoint(x,y,homoG):
    x1 = 0
    y1 = 0

    z = homoG[2,0] * x + homoG[2,1]*y + homoG[2,2]
    scale = 1.0/z
    x1 = cv.Round(homoG[0,0]*x + homoG[0,1]*y + homoG[0,2]*scale)
    y1 = cv.Round(homoG[1,0]*x + homoG[1,1]*y + homoG[1,2]*scale)
    #if ((x1<0) or (y1<0)):
        #print x1, y1
    return abs(x1), abs(y1)

def stitchImages(image1, image2, homography):

    
    
    #shiftX = int(abs(homography[0, 2]) * 2)
    #shiftY = int(abs(homography[1, 2]) * 2)
    
    #build a large canvas on which to paste our two images
    imgX = int(cv.Round((image1.width + image2.width) * 1.5))
    imgY = int(cv.Round((image1.height + image2.height) * 1.5))
    canvas = cv.CreateImage( (imgX,imgY), 8 ,3)



    #do histogram equalization on both images
    #image1 = equalize(image1)
    #image2 = equalize(image2)


    # first copy the starting (seed for homography) image to the canvas
    cv.SetImageROI(canvas, (0,0,image1.width, image1.height))
    cv.Copy(image1, canvas)
    cv.ResetImageROI(canvas)
    
    #cv.WarpPerspective(image1, canvas, homography, 16, (255,255,255,0))
    # save it to compare
    cv.SaveImage("debug.jpg",canvas) # for kevin's testing purposes

                                     #now warp in the second image on top
    cv.WarpPerspective(image2, canvas, homography, 16)

    return canvas  #return this new image



def compareSURFDescriptors(image1Descriptor, image2Descriptor, lastMinSquaredDistance):
    total = 0

    for i in range(len(image1Descriptor)):
        total+=math.pow((image1Descriptor[i]-image2Descriptor[i]),2)
        if (total  > lastMinSquaredDistance):
            break
        
    return math.sqrt(total)

def makeCVMatFromKeyPointList(list1):
    size = len(list1)
    arr = numpy.zeros((2,size))   # we want a simply 2d array. one column for x, the other for y
    for i in range(size):
        arr[0][i] = list1[i][0]
        arr[1][i] = list1[i][1]
    return arr

def findNaiveNearestNeighbor(image1Descriptor, image1KeyPoint, image2Descriptors, image2KeyPoints):
   descriptorsCount = len(image2Descriptors)
   
   minSquaredDistance = sys.float_info.max
   lastMinSquaredDistance = sys.float_info.max
   neighbor = 0
   
   for i in range(descriptorsCount):
      
      image2KeyPoint = image2KeyPoints[i]
      image2Descriptor = image2Descriptors[i]

      #laplacian is position 1
      if (image1KeyPoint[1] == image2KeyPoint[1]):
          squaredDistance = compareSURFDescriptors(image1Descriptor, image2Descriptor, lastMinSquaredDistance)
          if (squaredDistance < minSquaredDistance):
              neighbor = i
              lastMinSquaredDistance = minSquaredDistance
              minSquaredDistance = squaredDistance
          elif (squaredDistance < lastMinSquaredDistance):
              lastMinSquaredDistance = squaredDistance

   if (minSquaredDistance < 0.7 * lastMinSquaredDistance):
         return neighbor
   return NO_NEIGHBOR # -1

def cropWidth(maxWidth, imageIn):
    #copy the image passed in so we don't destroy it
    newImage = cv.CreateImage(cv.GetSize(imageIn), imageIn.depth, imageIn.nChannels)
    cv.Copy(imageIn, newImage)
    #set the region of interest on newimage
    cv.SetImageROI(newImage, (0, 0, maxWidth, newImage.height))
    #create destination image
    image2 = cv.CreateImage(cv.GetSize(newImage), newImage.depth, newImage.nChannels)
    #copy image over
    cv.Copy(newImage, image2, None)
    return newImage

def cropHeight(maxHeight, imageIn):
    #copy the image passed in so we don't destroy it
    newImage = cv.CreateImage(cv.GetSize(imageIn), imageIn.depth, imageIn.nChannels)
    cv.Copy(imageIn, newImage)
    #set the region of interest on newimage
    cv.SetImageROI(newImage, (0, 0, newImage.width, maxHeight))
    #create destination image
    image2 = cv.CreateImage(cv.GetSize(newImage), newImage.depth, newImage.nChannels)
    #copy image over
    cv.Copy(newImage, image2, None)
    return newImage

def crop(image):
    
    #get max width
    #loop through pixels starting at the far right
    #as soon as we find a color item, stop
    #maxwidth is that column + 1
    maxwidth = image.width
    for i in range((image.width - 1), 0, -1):
        foundNonBlack = 0
        for j in range(image.height -1):
            pixel = image[j,i]
            if (pixel[0] != 0) or (pixel[1] != 0) or (pixel[2] != 0):
                maxwidth = i + 1
                foundNonBlack = 1
        if foundNonBlack:
            break

    #get max height
    maxheight = image.height
    for i in range((image.height -1), 0, -1):
        foundNonBlack = 0
        for j in range(image.width -1):
            pixel = image[i,j]
            if (pixel[0] != 0) or (pixel[1] != 0) or (pixel[2] != 0):
               maxheight = i+1
               foundNonBlack = 1
        if foundNonBlack:
            break

    #crop along each axis
    image = cropWidth(maxwidth, image)
    image = cropHeight(maxheight, image)
    return image

def equalize(image1):
    #split image into channels
    redChan1 = cv.CreateImage(cv.GetSize(image1), 8, 1)
    greenChan1 = cv.CreateImage(cv.GetSize(image1), 8, 1)
    blueChan1 = cv.CreateImage(cv.GetSize(image1), 8 , 1)

    cv.Split(image1, blueChan1, greenChan1, redChan1, None)
    
    #run histogram equalization on each channel
    cv.EqualizeHist(redChan1, redChan1)
    cv.EqualizeHist(greenChan1, greenChan1)
    cv.EqualizeHist(blueChan1, blueChan1)
    
    #merge the color channels back together
    imgavg1 = cv.CreateImage(cv.GetSize(image1), 8, 3)
    cv.Merge(blueChan1, greenChan1, redChan1, None, image1)
    return image1

def normalize(image):
    redChan = cv.CreateImage(cv.GetSize(image), 8, 1)
    greenChan = cv.CreateImage(cv.GetSize(image), 8, 1)
    blueChan = cv.CreateImage(cv.GetSize(image), 8 , 1)

    redavg = cv.CreateImage(cv.GetSize(image), 8, 1)
    greenavg = cv.CreateImage(cv.GetSize(image), 8, 1)
    blueavg = cv.CreateImage(cv.GetSize(image), 8, 1)


    imgavg = cv.CreateImage(cv.GetSize(image), 8, 3)

    cv.Split(image, blueChan, greenChan, redChan, None)

    for x in range(image.width):
        for y in range(image.height):
            redVal = cv.GetReal2D(redChan, y, x)
            greenVal = cv.GetReal2D(greenChan, y, x)
            blueVal = cv.GetReal2D(blueChan, y, x)

            sum = redVal + greenVal + blueVal
            if sum==0:
                sum = .00001
            cv.SetReal2D(redavg, y, x, redVal/sum*255)
            cv.SetReal2D(greenavg, y, x, greenVal/sum*255)
            cv.SetReal2D(blueavg, y, x, blueVal/sum*255)
    cv.Merge(blueavg, greenavg, redavg, None, imgavg)
    return imgavg
    
def stitch(file1, file2):




    
    
    # load two our dataset images to test with
    print "Loading images..."
    cimg1 = equalize(cv.LoadImageM(file1, cv.CV_LOAD_IMAGE_COLOR))
    cimg2 = equalize(cv.LoadImageM(file2, cv.CV_LOAD_IMAGE_COLOR))
    img1 = cv.CreateImage(cv.GetSize(cimg1), 8, 1)
    img2 = cv.CreateImage(cv.GetSize(cimg1), 8, 1)
    cv.CvtColor(cimg1, img1, cv.CV_RGB2GRAY)
    cv.CvtColor(cimg2, img2, cv.CV_RGB2GRAY)
    print "Done."

    cv.NamedWindow("img1", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("img2", cv.CV_WINDOW_AUTOSIZE)
    sz = cv.GetSize(cimg1)
    cv.MoveWindow("img2", sz[0], 0)
    cv.ShowImage("img1", cimg1)
    cv.ShowImage("img2", cimg2)

    imgKP1 = cv.CreateImage(cv.GetSize(cimg1), 8, 3)
    imgKP2 = cv.CreateImage(cv.GetSize(cimg1), 8, 3)

    cv.Copy(cimg1, imgKP1, None)
    cv.Copy(cimg2, imgKP2, None)
    
    # Using cvExtractSURF
    # find some keypoints using SURF on our two files
    print "Extracting SURF keypoints..."
    hessianValue = 1000
    (img1_keypoints, img1_descriptors) = cv.ExtractSURF(img1, None, cv.CreateMemStorage(), (0, hessianValue, 3, 1))
    (img2_keypoints, img2_descriptors) = cv.ExtractSURF(img2, None, cv.CreateMemStorage(), (0, hessianValue, 3, 1))
    print "Done"

    for keyP in img1_keypoints:
        #print keyP
        cv.Circle(imgKP1, (int(keyP[0][0]), int(keyP[0][1])), 3, cv.RGB(255,0,255), thickness=-3, lineType=8, shift=0)

    for keyP in img2_keypoints:
        cv.Circle(imgKP2, (int(keyP[0][0]), int(keyP[0][1])), 3, cv.RGB(255,0,255), thickness=-3, lineType=8, shift=0)    

    cv.SaveImage("key1.jpg", imgKP1)
    cv.SaveImage("key2.jpg", imgKP2)

    cv.DestroyWindow("img1")
    cv.DestroyWindow("img2")
    cv.NamedWindow("img1", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("img2", cv.CV_WINDOW_AUTOSIZE)
    cv.MoveWindow("img2", sz[0], 0)
    cv.ShowImage("img1", imgKP1)
    cv.ShowImage("img2", imgKP2)
    
    #A SURF keypoint, represented as a tuple ((x, y), laplacian, size, dir, hessian) .

    # Matching Key Points in Both Images
    keyPointMatches1 = []
    keyPointMatches2 = []
    print "Running K Nearest Neighbor...    (this may take a while)"
    for i in range(len(img1_keypoints)):
        #do some KNN siliness
        nn = findNaiveNearestNeighbor(img1_descriptors[i], img1_keypoints[i], img2_descriptors, img2_keypoints)
        if (nn!=NO_NEIGHBOR):
            keyPointMatches1.append(img1_keypoints[i][0]) # append point tuple
            keyPointMatches2.append(img2_keypoints[nn][0]) # same
    print "Done."
    print len(keyPointMatches1)
    print " matches found"
    
    # Finding the Homography
    if (len(keyPointMatches1)>0):

        cv.Copy(cimg1, imgKP1, None)
        cv.Copy(cimg2, imgKP2, None)

        for i in range(len(keyPointMatches1)):
            keyP1 = (int(keyPointMatches1[i][0]), int(keyPointMatches1[i][1]))
            keyP2 = (int(keyPointMatches2[i][0]), int(keyPointMatches2[i][1]))
            cv.Circle(imgKP1, keyP1, 3, cv.RGB(0,255,0), thickness=-1, lineType=8, shift=0)
            cv.Circle(imgKP2, keyP2, 3, cv.RGB(0,255,0), thickness=-1, lineType=8, shift=0)

        cv.SaveImage("Mkey1.jpg", imgKP1)
        cv.SaveImage("Mkey2.jpg", imgKP2)

        cv.DestroyWindow("img1")
        cv.DestroyWindow("img2")
        cv.NamedWindow("img1", cv.CV_WINDOW_AUTOSIZE)
        cv.NamedWindow("img2", cv.CV_WINDOW_AUTOSIZE)
        cv.MoveWindow("img2", sz[0], 0)
        cv.ShowImage("img1", imgKP1)
        cv.ShowImage("img2", imgKP2)
        
        print "Finding the homography using RANSAC..."
        image1Points = cv.fromarray(makeCVMatFromKeyPointList(keyPointMatches1))
        image2Points = cv.fromarray(makeCVMatFromKeyPointList(keyPointMatches2))
        H = cv.fromarray(numpy.ones((3,3)))
        
        #print H
        cv.FindHomography(image1Points, image2Points, H, cv.CV_RANSAC, ransacReprojThreshold=1.0)
        #cv.GetPerspectiveTransform(keyPointMatches1[0:3], keyPointMatches2[0:3], H)
        #cv.FindFundamentalMat(image1Points, image2Points, H, method=cv.CV_FM_RANSAC, param1=1., param2=0.99)
        if (H != None):
            print "Done."
            print "Stitching images..."
            print H[0,0],H[0,1],H[0,2]
            print H[1,0],H[1,1],H[1,2]
            print H[2,0],H[2,1],H[2,2]
            #H[0:2,2]+=10
            #cv.Set2D(H,0,2, H[0,2]+100)
            #cv.Set2D(H,1,2, H[1,2]+100)
            #cv.Set2D(H,2,2, H[2,2]+100)
            newImg = stitchImages(cimg1,cimg2,H)

            #crop the image to remove all the black
            newImg = crop(newImg)

            cv.SaveImage("stitched.jpg", newImg)

            cv.NamedWindow("final")
            cv.ShowImage("final", newImg)
            cv.WaitKey(0)
            
        #        now glue images
            print "Done."
        else:
            print "FindHomography error"
    else:
        print "No matches found"




"""
Test progress bar code. Please ignore.
prog = ProgressBar(0, 100, 50, mode='fixed')
oldprog = str(prog)
for i in xrange(101):
    prog.update_amount(i)
    if oldprog != str(prog):
        print prog, "\r",
        sys.stdout.flush()
        oldprog=str(prog)
"""

#f2 = "100_3076.JPG"
#f1 = "100_3084.JPG"

f1 = "s100_2991.JPG"
f2 = "s100_2993.JPG"
stitch(f1,f2)
