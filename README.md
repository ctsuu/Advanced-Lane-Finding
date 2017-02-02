# Advanced-Lane-Finding
Udacity Self-Driving Car Engineer Class Project, due on Feb 13th, 2017



The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Any camera lens will have distortion problem as showing. I intent to use gopro camera for some future work, so I would like to use my own picture to give a try.  
<p align="center">
 <img src="./output_images/GOPR7216s.JPG" width="720">
</p>
The code for this step is contained in the first code cell of the IPython notebook located in "./Advanced-Lane_Finging.ipynb".
I am using opencv method. The key idea is using a knowing object, take many pictures from the camera and lens that you want to calibrate. You define the point location, you also need find point location in the picture space. Because of the lens distortion, the points location will not match at this point. Also do this manually is very time consuming and lead to error as well. OpenCV have automatic tools can help. 

I downloaded a 10x7 black and white chessboard from Udacity class for this task. I printed on 11x17 paper and put on flat surface. The chessboard has 9x6 inner corners. The black and white contrast make it easy and reliable detected by opencv. For the code, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard inner corners in the world. Here is 9x6, counted by hand. For this project, I am using a flat surface, the z will be always = 0.  Thus, for each test image iteration, the detected corner (x,y) pixel location will be appended to `imgpoint`, and knowning `objp` will be appended into `objpoint`. The process needs 20 - 30 images from different angles and distances to cover the protential usage of the camera. 

Then I can use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
<p align="center">
 <img src="./output_images/GOPR7216u.JPG" width="720">
</p>

Then, save the result for future use. 
```
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "output_images/720x540_pickle.p", "wb" ) )
```

###Pipeline (single images)

####1. Apply distortion-correction to raw image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
Simply take the saved distortion_correction matrix for the right camera and right resulotion. For speedup the calculation and display, I resized the image to 720x540. All calibration is done on 720 resolution. I forgot change the input image size, it will look like this: 
<p align="center">
 <img src="./output_images/what_happening.png" width="800">
</p>

```
# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "output_images/720x540_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
```
If do it right, the image will look like this:
<p align="center">
 <img src="./output_images/Undistorted_and_warped_2.png" width="800">
</p>


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. 
I defined each transform and methods as a function, so I can call them later in the pipeline. 
```
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output
```
The sobel x and sobel y behaive totally different, some test images x is better than y, some test is opposite. So I have to use both of them.  
<p align="center">
 <img src="./output_images/Thresholded_x_Gradient.png" width="800">
</p>

<p align="center">
 <img src="./output_images/Thresholded_y_Gradient.png" width="800">
</p>

```
# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
```

<p align="center">
 <img src="./output_images/Thresholded_mag_Gradient.png" width="800">
</p>

It looks close to Sobel y in this case. 

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

my perspective transform takes raw image (720) and transform matrix M, warp the image to bird view. the code is located in the  code cell of the IPython notebook. 
```
def perspective_transform(img, M):
    warped = cv2.warpPerspective(img, M, dsize = (img.shape[1],img.shape[0]), flags = cv2.INTER_LINEAR)
    return warped

```
The matrix M is generated by `cv2.getPerspectiveTransform(src, dst)`. My approach is defind the apex point first, the polybox is measured left and right offset by 50 pixels. The near bottom of the raw image, I take the offset from outside edge by 10 pixels., 
```
apex, apey = 360, 258
offset_far = 50
offset_near = 10
src = np.float32([[int(apex-offset_far),apey],
                  [int(apex+offset_far),apey],
                  [int(0+offset_near),390],
                  [int(720-offset_near),390]])
dst = np.float32([[0,0],[720,0],[0,405],[720,405]])
M = cv2.getPerspectiveTransform(src, dst)

#Calculate inverted matrix for future use:
Mi = cv2.getPerspectiveTransform(dst, src)

```
I verified that my perspective transform was working as expected for resized images. Load a test image and its warped counterpart to verify that the lane lines appear parallel in the warped image.

<p align="center">
 <img src="./output_images/Bird_view_transform.png" width="800">
</p>

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I sliced the input image into n= 8 bars, each bar, I calculation the histogram and find the two peaks can represent the lane line location. Then save the good location points for left and right lane separely, then feed into a 2nd order polynomial function as showing:
```
from scipy import signal

histogram = np.sum(combined[int(combined.shape[0]*3/8):int(combined.shape[0]*1/2),:], axis=0)
plt.plot(histogram)

peakind = signal.find_peaks_cwt(histogram, np.arange(1,15))
```
<p align="center">
 <img src="./output_images/histogram_and_find_Peaks.png" width="800">
</p>
There are two part of work: 
1. From last n sliced window, we can collect at least 3 good points for left or right lane, for example `lefty` and `leftx` that can fit into `left_fit = np.polyfit(lefty, leftx, 2)`, the output is the A, B, C values for polunomial function f(y)=Ay​2​​+By+C. 
2. Calculate the left_fitx and right_fitx values according to each row in the picture space. `yvals` is an array to match the rows count in pixel/picture space. 

```
yvals = np.linspace(0, img.shape[0], num=img.shape[0])
left_fit = np.polyfit(lefty, leftx, 2)
left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
right_fit = np.polyfit(righty, rightx, 2)
right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]
```
Then plot collected data points as red and blue dots, detected lane show as continuce curve:
<p align="center">
 <img src="./output_images/curve_fit.png" width="800">
</p>

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Remember, the "bird_view" in this pipeline is not true bird view. Because of the camera is pointing forward and downward at small angle, in the view, the vertical pixel may repersent 100m and further. The horizotal pixel is repersenting the car width closely. There are the factors I used to convert pixel space to true environment.
```
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/405 # meters per pixel in y dimension
xm_per_pix = 3.7/600 # meteres per pixel in x dimension
```

Pick a y-value where we want radius of curvature. I choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(yvals). Because it is close to the vehical, the curve can reflect the steering angles. 

```
left_fit_cr = np.polyfit(np.array(lefty,dtype=np.float32)*ym_per_pix, \
                         np.array(leftx,dtype=np.float32)*xm_per_pix, 2)
right_fit_cr = np.polyfit(np.array(righty,dtype=np.float32)*ym_per_pix, \
                          np.array(rightx,dtype=np.float32)*xm_per_pix, 2)

def left_curverad(left_fit_cr, y_eval):
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                             /np.absolute(2*left_fit_cr[0])
    return left_curverad

def right_curverad(right_fit_cr, y_eval):
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                                /np.absolute(2*right_fit_cr[0])
    return right_curverad
```


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
