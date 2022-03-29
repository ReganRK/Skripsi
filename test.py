from PIL import Image
import os
import json
import cv2
import numpy
import colorgram

#get all file name
allImageFiles = os.listdir("AnimeDataset/data/train/")
allColorgramFiles = os.listdir("AnimeDataset/data/colorgram/")

img = Image.open("AnimeDataset/data/train/" + allImageFiles[0]).convert('RGB')
# Crop the image using crop() method
ColorImage = img.crop((0, 0, 512,512))
SketchImage = img.crop((512, 0, 1024, 512))
ColorImage.save("RealData/Color/" + allImageFiles[0])
SketchImage.save("RealData/Sketch/" + allImageFiles[0])
ColorImage.show()
SketchImage.show()
# print(allColorgramFiles[0])
img.show()

color_image = cv2.imread("RealData/Color/" + allImageFiles[0])
sketch_image = cv2.imread("RealData/Sketch/" + allImageFiles[0])

# converting BGR to RGB
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
# sketch_image = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2RGB)

cv2.imshow("color_image", color_image)
cv2.imshow("sketch_image", sketch_image)
cv2.waitKey()

(height, width) = color_image.shape[:2]
print(height)
print(width)

pojok_kiri_atas = color_image[0:256, 0:256]
pojok_kiri_bawah = color_image[256:512, 0:256]
pojok_kanan_atas = color_image[0:256, 256:512]
pojok_kanan_bawah = color_image[256:512, 256:512]

# pojok_kiri_atas = cv2.cvtColor(pojok_kiri_atas, cv2.COLOR_BGR2RGB)
# pojok_kiri_bawah = cv2.cvtColor(pojok_kiri_bawah, cv2.COLOR_BGR2RGB)
# pojok_kanan_atas = cv2.cvtColor(pojok_kanan_atas, cv2.COLOR_BGR2RGB)
# pojok_kanan_bawah = cv2.cvtColor(pojok_kanan_bawah, cv2.COLOR_BGR2RGB)

# cv2.imshow("pojok_kiri_atas", pojok_kiri_atas)
# cv2.imshow("pojok_kiri_bawah", pojok_kiri_bawah)
# cv2.imshow("pojok_kanan_atas", pojok_kanan_atas)
# cv2.imshow("pojok_kanan_bawah", pojok_kanan_bawah)
# cv2.waitKey()

# tempat_taruh_colorgram = []

# Opening JSON file
# f = open("AnimeDataset/data/colorgram/" + allColorgramFiles[0])

# returns JSON object as
# a dictionary
# data = json.load(f)

# for saiki in data.values():
#     for dalem_saiki in saiki.values():
#         tempat_taruh_colorgram.append(dalem_saiki)

# f.close()

# print(tempat_taruh_colorgram)

colors = colorgram.extract("RealData/Color/" + allImageFiles[0], 10)
pojok_kiri_atas = numpy.array(pojok_kiri_atas)
pojok_kiri_bawah = numpy.array(pojok_kiri_bawah)
pojok_kanan_atas = numpy.array(pojok_kanan_atas)
pojok_kanan_bawah = numpy.array(pojok_kanan_bawah)

# print(colors)

# Iki seng buat pojok kiri atas
for colors_saiki in colors:
    # print(colors_saiki[0])
    red = colors_saiki.rgb.r
    green = colors_saiki.rgb.g
    blue = colors_saiki.rgb.b

    temp_color_check = [red, green, blue]

    indices = numpy.where(pojok_kiri_atas == temp_color_check)
    coordinates = zip(indices[0], indices[1])
    unique_coordinates = list(set(list(coordinates)))

    panjange = len(unique_coordinates) / 2

    # if panjange != 0:
    #     # print(unique_coordinates[int(panjange)])
    #     this_is_the_spot_x = unique_coordinates[int(panjange)][0]
    #     this_is_the_spot_y = unique_coordinates[int(panjange)][1]
    #     # print(this_is_the_spot_x)
    #     # print(this_is_the_spot_y)
    #     sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
    #                               color=(blue, green, red), thickness=-1)

    for pixel_saiki in unique_coordinates:
        this_is_the_spot_x = pixel_saiki[0]
        this_is_the_spot_y = pixel_saiki[1]

        sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
                                  color=(blue, green, red), thickness=-1)
#
# iki seng buat pojok kanan atas
for colors_saiki in colors:
    # print(colors_saiki[0])
    red = colors_saiki.rgb.r
    green = colors_saiki.rgb.g
    blue = colors_saiki.rgb.b

    temp_color_check = [red, green, blue]

    indices = numpy.where(pojok_kanan_atas == temp_color_check)
    coordinates = zip(indices[0], indices[1])
    unique_coordinates = list(set(list(coordinates)))

    panjange = len(unique_coordinates) / 2

    if panjange != 0:
        # print(unique_coordinates[int(panjange)])
        this_is_the_spot_x = unique_coordinates[int(panjange)][0]
        this_is_the_spot_y = unique_coordinates[int(panjange)][1]
        this_is_the_spot_y+=256
        # print(this_is_the_spot_x)
        # print(this_is_the_spot_y)
        sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
                                  color=(blue, green, red), thickness=-1)

    # for pixel_saiki in unique_coordinates:
    #     this_is_the_spot_x = pixel_saiki[0]
    #     this_is_the_spot_y = pixel_saiki[1]
    #     this_is_the_spot_y += 256
    #     sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
    #                               color=(blue, green, red), thickness=-1)

# iki seng buat kiri bawah
for colors_saiki in colors:
    # print(colors_saiki[0])
    red = colors_saiki.rgb.r
    green = colors_saiki.rgb.g
    blue = colors_saiki.rgb.b

    temp_color_check = [red, green, blue]

    indices = numpy.where(pojok_kiri_bawah == temp_color_check)
    coordinates = zip(indices[0], indices[1])
    unique_coordinates = list(set(list(coordinates)))

    panjange = len(unique_coordinates) / 2

    if panjange != 0:
        # print(unique_coordinates[int(panjange)])
        this_is_the_spot_x = unique_coordinates[int(panjange)][0]
        this_is_the_spot_y = unique_coordinates[int(panjange)][1]
        this_is_the_spot_x += 256
        # print(this_is_the_spot_x)
        # print(this_is_the_spot_y)
        sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
                                    color=(blue, green, red), thickness=-1)
#
# iki seng buat kanan bawah
for colors_saiki in colors:
    # print(colors_saiki[0])
    red = colors_saiki.rgb.r
    green = colors_saiki.rgb.g
    blue = colors_saiki.rgb.b

    temp_color_check = [red, green, blue]

    indices = numpy.where(pojok_kanan_bawah == temp_color_check)
    coordinates = zip(indices[0], indices[1])
    unique_coordinates = list(set(list(coordinates)))

    panjange = len(unique_coordinates) / 2

    if panjange != 0:
        # print(unique_coordinates[int(panjange)])
        this_is_the_spot_x = unique_coordinates[int(panjange)][0]
        this_is_the_spot_y = unique_coordinates[int(panjange)][1]
        this_is_the_spot_x += 256
        this_is_the_spot_y += 256
        # print(this_is_the_spot_x)
        # print(this_is_the_spot_y)
        sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
                                    color=(blue, green, red), thickness=-1)
#
#
# # save image
status = cv2.imwrite(allImageFiles[0], sketch_image)
cv2.imshow("Hasil", sketch_image)
cv2.waitKey()
# print("Image written to file-system : ", status)