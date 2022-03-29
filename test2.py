# import skimage
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
import colorgram
import natsort
# from matplotlib import pyplot as plt

# import colorsys
# Importing required libraries
# from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
# from skimage.data import astronaut
from skimage.color import label2rgb
# import skimage.future.graph as graph
# import skimage.measure as measure
from skimage import io

#get all file name
allImageFiles = os.listdir("AnimeDataset/data/train/")
allColorgramFiles = os.listdir("AnimeDataset/data/colorgram/")

img = Image.open("AnimeDataset/data/train/" + allImageFiles[0]).convert('RGB')

# Crop the image using crop() method
ColorImage = img.crop((0, 0, 512,512))
SketchImage = img.crop((512, 0, 1024, 512))
ColorImage.save("RealData/Color/" + allImageFiles[0])
SketchImage.save("RealData/Sketch/" + allImageFiles[0])
# ColorImage.show()
# SketchImage.show()
# img.show()

color_image = Image.open("RealData/Color/" + allImageFiles[0]).convert('RGB')
sketch_image = Image.open("RealData/Sketch/" + allImageFiles[0]).convert('RGB')

color_image = np.array(color_image)
sketch_image = np.array(sketch_image)

# color_image = cv2.imread("RealData/Color/" + allImageFiles[0])
# sketch_image = cv2.imread("RealData/Sketch/" + allImageFiles[0])

# converting BGR to RGB
# color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
# sketch_image_rgb = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2RGB)

# cv2.imshow("Color Image", color_image)
# cv2.waitKey(0)

color_image_segments = felzenszwalb(color_image, 100)

color_image_segments_colored = label2rgb(color_image_segments, color_image, kind='avg')

# color_image_segments_colored.save("RealData/Region/" + allImageFiles[0])

io.imsave("RealData/Region/" + allImageFiles[0], color_image_segments_colored)

this_is_the_region = Image.open("RealData/Region/" + allImageFiles[0]).convert('RGB')

this_is_the_region.show()

# this_is_the_region = cv2.imread("RealData/Region/" + allImageFiles[0])
#
# cv2.imshow("Region Image", this_is_the_region)
# cv2.waitKey(0)

# width, height = this_is_the_region.size
#
# print(width)
# print(height)

kelipatan = 128
nomer = 0

for kebawah in range(4):
    for kesamping in range(4):
        satu = kelipatan*kesamping
        dua = kelipatan*kebawah
        tiga = kelipatan*(kesamping+1)
        empat = kelipatan*(kebawah+1)
        # print(satu)
        # print(dua)
        # print(tiga)
        # print(empat)
        cropped_this_is_the_region = this_is_the_region.crop((satu, dua, tiga, empat))
        nomer += 1
        # print(nomer)
        # cropped_this_is_the_region.show()
        cropped_this_is_the_region.save("RealData/Cropped_Region/" + str(nomer) + ".png")

# opened_Cropped_Image = cv2.imread("RealData/Region/" + allCroppedRegionFileName[0])

kekanan = 0
kebawah = 0

for saiki in range(16):
    namaFile_e = "RealData/Cropped_Region/" + str(saiki+1) + ".png"
    colors = colorgram.extract(namaFile_e, 10)
    opened_Cropped_Image = cv2.imread(namaFile_e)

    for color_saiki in colors:
        red = color_saiki.rgb.r
        green = color_saiki.rgb.g
        blue = color_saiki.rgb.b

        if red == 0 & green == 0 & blue == 0:
            continue

        temp_color_check = [blue, green, red]
        indices = np.where(temp_color_check == opened_Cropped_Image)
        coordinates = zip(indices[0], indices[1])
        unique_coordinates = list(set(list(coordinates)))

        panjange = len(unique_coordinates) / 2

        for pixel_saiki in unique_coordinates:
            this_is_the_spot_x = pixel_saiki[0]
            this_is_the_spot_y = pixel_saiki[1]
            this_is_the_spot_x = this_is_the_spot_x + ((kekanan % 4) * kelipatan)
            this_is_the_spot_y = this_is_the_spot_y + (kebawah * kelipatan)
            sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
                                      color=(blue, green, red), thickness=-1)

        # if panjange != 0:
        #     # print(unique_coordinates[int(panjange)])
        #     this_is_the_spot_x = unique_coordinates[int(panjange)][0]
        #     this_is_the_spot_y = unique_coordinates[int(panjange)][1]
        #     this_is_the_spot_x = this_is_the_spot_x + (kekanan * kelipatan)
        #     this_is_the_spot_y = this_is_the_spot_y + (kebawah * kelipatan)
        #     # print(this_is_the_spot_x)
        #     # print(this_is_the_spot_y)
        #     sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
        #                               color=(blue, green, red), thickness=-1)
    kekanan = kekanan + 1
    if kekanan % 4 == 0:
        kebawah = kebawah + 1

    print("nama file")
    print(namaFile_e)
    print("kekanan")
    print(kekanan % 4)
    print("kebawah")
    print(kebawah)

# save image
status = cv2.imwrite(allImageFiles[0], sketch_image)
cv2.imshow("Hasil", sketch_image)
cv2.waitKey()
print("Image written to file-system : ", status)

# array_opening_croped_label_image = np.array(opening_croped_label_image)

# print(array_opening_croped_label_image.size)

# print(colors)

# cv2.imshow("nyobak", color_image_segments_colored)
# cv2.waitKey(0)

# tampilin_bentar = color_image_segments_colored[0:64, 0:64]
# cv2.imshow("nyobak", tampilin_bentar)
# cv2.waitKey(0)

# extract all colors from colored image
# colors = colorgram.extract("RealData/Region/" + allImageFiles[0], 10)

# print(colors)