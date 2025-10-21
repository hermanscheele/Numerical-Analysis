from skimage import img_as_float, color
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from utils import svd, plot_log_singular_vals

# Read images
im1 = iio.imread("img/board-157165_1280.png")
im2 = iio.imread("img/jellyfish-698521_1280.jpg")
im3 = iio.imread("img/outdoors-5129182_1280.jpg")

# Convert to greyscale (2D)
im1_grey = color.rgb2gray(im1)
im2_grey = color.rgb2gray(im2)
im3_grey = color.rgb2gray(im3)

# Convert to double (0, 1)
im1 = img_as_float(im1_grey)
im2 = img_as_float(im2_grey)
im3 = img_as_float(im3_grey)
ims = [('Chessboard', im1), ('Jellyfish', im2), ('City', im3)]


r = 50
U, S, V_T = svd(im3)

sigmas = np.diag(S)[:r]

u = U[:, :r]
s = np.diag(sigmas)
v = V_T.T[:, :r]

im = u @ s @ v.T



plot_log_singular_vals(ims)



# plt.figure()
# plt.axis(False)
# plt.imshow(im, cmap='gray')
# plt.show()





