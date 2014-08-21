import numpy as np
from scipy import ndimage
from scipy import optimize

from skimage.morphology import watershed
from skimage.feature import peak_local_max

from pylab import close, colorbar, figure, imshow, show

def isoThreshold(imageData):
    def computeCriterion(threshold):
        objSeg = imageData > threshold
        objAve = imageData[objSeg].mean()
        bgAve = imageData[np.logical_not(objSeg)].mean()
        return (objAve + bgAve) / 2

    for t in range(256):
        criteria = computeCriterion(t)
        if t > criteria:
            break
    objSeg = imageData > t
    return objSeg

def doWSSeg(binaryImage):
    distance = ndimage.distance_transform_edt(binaryImage)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((5, 5)),labels=binaryImage)
    markers = ndimage.label(local_maxi)
    labels_ws = watershed(-distance, markers[0], mask=binaryImage)
    return labels_ws

def measure_area(image, mask):
    return (1*(mask==1)).sum()

def measure_mean(image, mask):
    return image[mask].mean()

def measure_std(image, mask):
    return image[mask].std()

def measure_areaFraction(image, mask):
    return (1*(mask==1)).mean()

measureNames = ["area", "mean", "std", "areaFraction"]

def doMeasures(image, imageLabels):
    measureResults = []

    for n, objSlice in enumerate(ndimage.find_objects(imageLabels)):
        imagePatch = image[objSlice]
        imageMask  = imageLabels[objSlice] == (n + 1)
        imageMeasures = []
        for measureName in measureNames:
            measureFunction = globals()["measure_" + measureName]
            measurement = measureFunction(imagePatch, imageMask)
            imageMeasures.append(measurement)

        measureResults.append(imageMeasures)
    return np.array(measureResults)

def doAnalysis(imageData):
    # Convert to a greyscale image
    greyScale = imageData.mean(axis=2)

    # Apply 1.5px gaussian blur
    blurredData = ndimage.gaussian_filter(greyScale, 1.5)

    # Do threshold (based on fiji "default" mode)
    thresholded = np.logical_not(isoThreshold(blurredData))

    # Label regions
    regions = doWSSeg(thresholded)

    # Compute per/region statistics
    measures = doMeasures(imageData, regions)

    # Compute the mean measure for all regions in this image
    return measures.mean(axis=0)


if __name__ == "__main__":
    import os
    folder = "Original"
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder)])

    for n, fileName in enumerate(files):
        print(n, fileName, doAnalysis(ndimage.imread(fileName)))
