import os
import numpy as np
from numpy import asarray
from PIL import Image
from matplotlib import image as mpimg
from matplotlib import pyplot

if not os.path.exists('opera_house.jpg'):
    dummy = np.random.randint(0, 255, (186, 931, 3), dtype=np.uint8)
    Image.fromarray(dummy).save('opera_house.jpg')

data = mpimg.imread('opera_house.jpg')
print(data.dtype)
print(data.shape)
pyplot.imshow(data)
pyplot.show()

img = Image.open('opera_house.jpg')
new_image = img.resize((500, 500))
new_image.save('myimage_500.jpg')
pyplot.imshow(new_image)
pyplot.show()

data = asarray(img)
print(data.shape)
image2 = Image.fromarray(data)
print(image2.format)
print(image2.mode)
print(image2.size)

hoz_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
ver_flip = img.transpose(Image.FLIP_TOP_BOTTOM)

pyplot.subplot(311)
pyplot.imshow(img)
pyplot.subplot(312)
pyplot.imshow(hoz_flip)
pyplot.subplot(313)
pyplot.imshow(ver_flip)
pyplot.show()

pyplot.subplot(311)
pyplot.imshow(img)
pyplot.subplot(312)
pyplot.imshow(img.rotate(45))
pyplot.subplot(313)
pyplot.imshow(img.rotate(90))
pyplot.show()
