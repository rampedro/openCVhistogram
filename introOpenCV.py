import cv2
print(cv2.__version__)



import urllib.request
imageUrl = "https://scontent.fykz1-1.fna.fbcdn.net/v/t1.0-9/71444_4991189531516_2048190561_n.jpg?_nc_cat=105&ccb=2&_nc_sid=f9d7a1&_nc_ohc=6NqSfiUJMy0AX-1l8lM&_nc_ht=scontent.fykz1-1.fna&oh=433fc5de95b625b02309bb2ed9b6e8d8&oe=601C9544"
imageName = "pedram.jpg"
urllib.request.urlretrieve(imageUrl,imageName)


###########################################################
# Does the file exist?
###########################################################

import os
print("Does the file {} exist: {}".format(imageName,imageName in os.listdir(os.curdir)))



from matplotlib import pyplot as plot
theImage = cv2.imread(imageName)

# Up until here the image is bluish

###########################################################
# Lets fix the colors
###########################################################

img_color_fixed = cv2.cvtColor(theImage,cv2.COLOR_BGR2RGB)
img_gray_fixed = cv2.cvtColor(theImage, cv2.COLOR_BGR2GRAY)

###########################################################
# Changing the displying image, not the image size itself #
###########################################################

#from pylab import rcParams
#rcParams['figure.figsize'] = 10,12


# subplot(n,m,t):  nxm grid the tth one

plot.subplot(3, 3, 1)
plot.imshow(img_color_fixed)
plot.title("Color",fontsize=6)
plot.axis("off") #remove axes ticks


plot.subplot(3, 3, 2)
plot.imshow(img_gray_fixed, 'gray')
plot.title("Gray",fontsize=6)
plot.axis("off") #remove axes ticks




###########################################################
# Canny Edge Detection #
###########################################################


#A simple edge detector using gradient magnitude
#1-Compute gradient vector at each pixel by
#convolving image with horizontal and
#vertical derivative filters
#2-Compute gradient magnitude at each pixel
#3-If magnitude at a pixel exceeds a threshold,
#report a possible edge point.

#MORE INFORMATION @ https://docs.opencv.org/master/da/d22/tutorial_py_canny.html?cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork-19816089&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork-19816089&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork-19816089&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork-19816089&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ

#To decide which are all edges are really edges and which are not. For this, we need two threshold values, minVal and maxVal. Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal are sure to be non-edges, so discarded. Those who lie between these two thresholds are classified edges or non-edges based on their connectivity. If they are connected to "sure-edge" pixels, they are considered to be part of edges. Otherwise, they are also discarded

edges = cv2.Canny(img_gray_fixed,
                  threshold1=70,
                  threshold2=150,apertureSize=3,L2gradient=True)

#L2gradient : Edge_Gradient(G)=sqrt(G2x+G2y) True
plot.subplot(3,3,3)
plot.imshow(edges,cmap = 'gray')
plot.title("Edges Gradient sqrt",fontsize=6)
plot.axis("off") #remove axes ticks



edges = cv2.Canny(img_gray_fixed,
                  threshold1=70,
                  threshold2=150,apertureSize=3,L2gradient=False)

#L2gradient : Edge_Gradient(G)=|Gx|+|Gy| Defult False
plot.subplot(3,3,4)
plot.imshow(edges,cmap = 'gray')
plot.title("Edges Gradient sums",fontsize=6)
plot.axis("off") #remove axes ticks


############################################################
# Generating the histogram of the gray scaled image #
############################################################

#Like any other histogram, this graph shows the count of something. In this case, it's the number of pixels in the image that are either dark (towards the left of the graph) or lighter (towards the right of the graph).
#darker pixels are on the left, and whiter pixels are to the right.
#Use-Case : to determin if your image is over or under exposed. by looking at a histogram that most of the values are shifted toward LEFT (larger numbers) we can conclude that the image is overexposed.

plot.subplot(3,3,5)
plot.hist(img_gray_fixed.ravel(),256,[0,256])
plot.title("Histogram of Gray scaled",fontsize=6)
plot.xticks(fontsize=6)
plot.yticks(fontsize=6)



plot.show()





















