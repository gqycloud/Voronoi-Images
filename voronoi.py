import numpy as np
import cv2 as cv2

def main():  
    print('Reading image...')
    # Read in the image.
    img = cv2.imread('testImage.jpg')
    
    # Generate SURF image
    surfSubdiv2 = getSurfSubdiv2(img)
    surfVoronoi = drawVoronoi(img, surfSubdiv2)
    
    #Generate FAST image
    fastSubdiv2 = getFastSubdiv2(img)
    fastVoronoi = drawVoronoi(img, fastSubdiv2)
    
    #Generate BRIEF image
    briefSubdiv2 = getBRIEFSubdiv2(img)
    briefVoronoi = drawVoronoi(img, briefSubdiv2)
    
    #Generate BRIEF image
    orbSubdiv2 = getORBSubdiv2(img)
    orbVoronoi = drawVoronoi(img, orbSubdiv2)
    
    print('Finished drawing voronoi...')
    cv2.imwrite('SURF.jpg',surfVoronoi)
    cv2.imwrite('FAST.jpg',fastVoronoi)
    cv2.imwrite('BRIEF.jpg',briefVoronoi)
    cv2.imwrite('ORB.jpg',orbVoronoi)


def drawVoronoi(img, subdiv): 
    voronoi = np.zeros(img.shape, dtype = img.dtype)
    ( facets, centers) = subdiv.getVoronoiFacetList([])
 
    for facetsIndex in range(0,len(facets)):
        # Generate array of polygon corners
        facetArray = []
        for facet in facets[facetsIndex] :
            facetArray.append(facet)
        
        # Get average color of polygon from original image
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, np.int32([facetArray]), (255,255,255));
        color = cv2.mean(img, mask)

        # Fill polygon with average color
        intFacet = np.array(facetArray, np.int)
        cv2.fillConvexPoly(voronoi, intFacet, color);
        
        # Draw lines around polygon
        polyFacets = np.array([intFacet])
        cv2.polylines(voronoi, polyFacets, True, (255, 255, 255), 1, cv2.LINE_AA, 0) 
    
    return voronoi

def getSurfSubdiv2(img):
    hessianThreshold = 1000
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold)
    surf.setUpright(True)
    kp, des = surf.detectAndCompute(img,None)
    keyPoints = cv2.KeyPoint_convert(kp)
    points = []
    for keyPoint in keyPoints: 
        points.append((keyPoint[0], keyPoint[1]))
    
    size = img.shape
    subdiv2DShape = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(subdiv2DShape);
    for p in points :
        subdiv.insert(p)
    
    return subdiv

def getFastSubdiv2(img): 
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, 1000)
    keyPoints = cv2.KeyPoint_convert(kp)
    points = []
    for keyPoint in keyPoints: 
        points.append((keyPoint[0], keyPoint[1]))
    
    size = img.shape
    subdiv2DShape = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(subdiv2DShape);
    for p in points :
        subdiv.insert(p)
    
    return subdiv

def getBRIEFSubdiv2(img): 
    star = cv2.xfeatures2d.StarDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(img,None)
    kp, des = brief.compute(img, kp)
    keyPoints = cv2.KeyPoint_convert(kp)
    points = []
    for keyPoint in keyPoints: 
        points.append((keyPoint[0], keyPoint[1]))
    
    size = img.shape
    subdiv2DShape = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(subdiv2DShape);
    for p in points :
        subdiv.insert(p)
    
    return subdiv

def getORBSubdiv2(img): 
    orb = cv2.ORB_create(1000)
    kp = orb.detect(img,None)
    keyPoints = cv2.KeyPoint_convert(kp)
    points = []
    for keyPoint in keyPoints: 
        points.append((keyPoint[0], keyPoint[1]))
    
    size = img.shape
    subdiv2DShape = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(subdiv2DShape);
    for p in points :
        subdiv.insert(p)
    
    return subdiv

if __name__ == '__main__':
    main()