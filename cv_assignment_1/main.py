import numpy
import numpy as np
import cv2 as cv

def jsdcalculation(query_image, support_image):
    # This is dkl and jsd calculation function.
    dkl_left_value = (query_image * np.log(query_image/((query_image+support_image)/2))).sum()
    dkl_right_value = (support_image * np.log(support_image/((query_image+support_image)/2))).sum()
    result_value = 0.5 * dkl_left_value + 0.5 * dkl_right_value
    return result_value

def l1Normalization(histogram):
    #This function gets the histogram list and then returned the n1 normalized list
    norm_val = np.linalg.norm(histogram, 1)
    error_margin = 10 ** -6
    res_value = histogram / norm_val + error_margin
    return res_value


def perChannelHistogram(image, blue_list,green_list,red_list, num_bin):
        # This function gets the rgb values and number of bins and then do the calculations
        # according to per channel histogram.
        blue_hist = np.zeros(num_bin,int)
        green_hist = np.zeros(num_bin,int)
        red_hist = np.zeros(num_bin,int)
        reshaped_Image = image.reshape(image.shape[0] * image.shape[0], 3)
        for j in range(num_bin):
            blue_hist[j] += (reshaped_Image[:, 0] // int(256/num_bin) == j).sum()
            green_hist[j] += (reshaped_Image[:, 1] // int(256 / num_bin) == j).sum()
            red_hist[j] += (reshaped_Image[:, 2] // int(256 / num_bin) == j).sum()
        blue_hist = l1Normalization(blue_hist)
        green_hist = l1Normalization(green_hist)
        red_hist = l1Normalization(red_hist)
        blue_list.append(blue_hist)
        green_list.append(green_hist)
        red_list.append(red_hist)


def per3DHistogram(image, hist_list_3d, num_bin):
    #This functions get the image histogram list and number of bins and do calculations
    # according to 3d histogram.
    quantitization = 256 // num_bin
    hist_list = np.zeros(num_bin ** 3,int)
    reshaped_Image = image.reshape( image.shape[0] * image.shape[0], 3)
    tmp_np = reshaped_Image[:, 0] // quantitization * (num_bin * num_bin) + reshaped_Image[:, 1] // quantitization * num_bin + reshaped_Image[:, 2] // quantitization
    unique, counts = numpy.unique(tmp_np, return_counts=True)
    hist_list[unique] = counts
    hist_list = l1Normalization(hist_list)
    hist_list_3d.append(hist_list)


def main(number_of_bin,is_it_per_channel,grid_num,query_name):
    #Create the support histogram list in order to use in the perchannel part.
    histogram_support96_blue = []
    histogram_support96_green = []
    histogram_support96_red = []
    # The lists are will be going to use in the perchannel part for the query list.
    histogram_queryx_blue = []
    histogram_queryx_green = []
    histogram_queryx_red = []
    #This is the lists for 3d histogram both query and support.
    histogram_3d_query = []
    histogram_3d_support96 = []
    # take the images input to the list
    with open('InstanceNames.txt') as f:
        all_instances = f.readlines()
        f.close()
    # this for loop for query 1
    for line in all_instances:
        # arrange the new line character and adjust the parameters
        if line[-1] == '\n':
            str_support = "support_96/" + line[:-1]
            str_queryx = query_name + line[:-1]
        else:
            str_support = "support_96/" + line
            str_queryx = query_name + line
        input_image_support = cv.imread(str_support)
        input_image_queryx = cv.imread(str_queryx)
        # check is it per channel or not
        if is_it_per_channel == True:
            if grid_num == 1:
                perChannelHistogram(input_image_support, histogram_support96_blue, histogram_support96_green,histogram_support96_red,number_of_bin)
                perChannelHistogram(input_image_queryx, histogram_queryx_blue, histogram_queryx_green,histogram_queryx_red, number_of_bin)
            else:
                # do grid calculations here
                interval = 96//grid_num
                for i in range(grid_num):
                    for j in range(grid_num):
                        perChannelHistogram(input_image_support[i * interval:(i+1)*interval,j * interval:(j+1) * interval], histogram_support96_blue, histogram_support96_green,histogram_support96_red, number_of_bin)
                        perChannelHistogram(input_image_queryx[i * interval:(i+1)*interval,j * interval:(j+1) * interval], histogram_queryx_blue, histogram_queryx_green, histogram_queryx_red, number_of_bin)
                        #cell = image[i * size:(i + 1) * size, j * size:(j + 1) * size]

        else:
            # check grid size
            if grid_num == 1:
                per3DHistogram(input_image_support, histogram_3d_support96, number_of_bin)
                per3DHistogram(input_image_queryx, histogram_3d_query ,number_of_bin)
            else:
                interval = 96 // grid_num
                for i in range(grid_num):
                    for j in range(grid_num):
                        per3DHistogram(input_image_support[i * interval:(i+1)*interval,j * interval:(j+1) * interval], histogram_3d_support96, number_of_bin)
                        per3DHistogram(input_image_queryx[i * interval:(i+1)*interval,j * interval:(j+1) * interval], histogram_3d_query, number_of_bin)
    # for query1 calculate every value
    if is_it_per_channel == True:
        grid_square = grid_num * grid_num
        right_guess = 0
        for i in range(0, len(histogram_queryx_blue)//grid_square):
            min_val = 99999
            min_index = -1
            for j in range(0, len(histogram_support96_blue)//grid_square):
                count = 0
                for k in range(0, grid_num * grid_num):
                    #do calculations and get the score
                    count += jsdcalculation(histogram_queryx_blue[grid_square*i+k], histogram_support96_blue[grid_square*j+k])
                    count += jsdcalculation(histogram_queryx_green[grid_square*i+k], histogram_support96_green[grid_square*j+k])
                    count += jsdcalculation(histogram_queryx_red[grid_square*i+k], histogram_support96_red[grid_square*j+k])
                if count < min_val:
                    # find min divergence
                    min_val = count
                    min_index = j
            if i == min_index:
                right_guess += 1
        print(right_guess/200)
    else:
        grid_square = grid_num * grid_num
        right_guess = 0
        for i in range(0, len(histogram_3d_query)//grid_square):
            min_val = 99999
            min_index = -1
            for j in range(0, len(histogram_3d_support96)//grid_square):
                count = 0
                for k in range(0, grid_num * grid_num):
                    # do calculations for the divergence value
                    count += jsdcalculation(histogram_3d_query[grid_square*i+k], histogram_3d_support96[grid_square*j+k])
                if count < min_val:
                    #find the minimum value
                    min_val = count
                    min_index = j
            if i == min_index:
                right_guess += 1
        print(right_guess/200)

# Here is the main function and I adjust the parameters here
# The first parameter define number of bins, second defines is it per channel or not
# Third parameter defines grid number and the last parameter defines the which query string
# Then the main function will print the accuracy result
if __name__ == '__main__':
    main(4, False, 4, "query_3/")
