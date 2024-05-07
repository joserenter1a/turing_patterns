import cv2 as cv 
import numpy as np

# This function just reads image name as input and returns the file path
def read_path_init():
    fpath = input(f"Enter the image name, \n(Make sure it is in the same folder as this program) \nExample: image.jpeg or image.png: ")
    return f'./{fpath}'

# this function creates our opencv image and resizes it to 600x800
def init_and_resize(fpath):
    img = cv.imread(fpath, 1)
    img = cv.resize(img, (600, 800))
    return img

# this function shows the image and allows you to press a key to close
def show_img(img):
    cv.imshow('TuringPatterns', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# this function acts as our 'reaction' or our sharpening function which
# just takes the kernel as a np array and returns the sharpened image
def reaction(img):
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                   [-1, -1, -1]])
    sharpened = cv.filter2D(img, -1, kernel_sharpening)
    return sharpened

# this is our 'diffusion' function or a blurring function which utilizes an
# openCV blur to blur the image
def diffusion(img):
    blurred = cv.stackBlur(img, (5, 5))
    return blurred
# this is our in between function which happens in between our diffusion and reaction
# in this case it modifies the brightness and contrast
def in_between(img):
    alpha = 1
    beta = -5
    return cv.filter(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)

# This is where our turing processing actually happens
# It takes in the image and an integer x, 
# and then peforms the diffusion reaction on our image x times
def turing(img, x: int):
    for i in range(x):
        # blur and sharpen
        img = reaction(img)
        #img = in_between(img)
        img = diffusion(img)
        

    # return the image
    return img

if __name__=='__main__':
    # read the image and initialize the path
    image_path = read_path_init()
    # initialize the image
    img = init_and_resize(image_path)
    # create our turing pattern
    t = turing(img, 100)
    # show both images, press any key to close
    show_img(img)
    show_img(t)