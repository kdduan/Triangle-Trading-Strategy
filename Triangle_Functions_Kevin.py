# Pennant Triangle Project
# By: Kevin Duan
# Fall 2019

from math import pi
import os
import shutil
import csv
import cv2
import glob
from bokeh.io import export_png
import pandas as pd
import datetime
from datetime import timedelta
import sys
from matplotlib import pyplot as plt
from statistics import mean
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import load_model
import pickle
import time
from bokeh.plotting import figure, show, output_file

FILE_DIRECTORY = r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\Market Database'
CANDLE_PNG_DIRECTORY = r'C:\Users\duans\Documents\1. Projects\3. Triangles\Fake Data CNN\Candlesticks'
DATA_DIR = r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\Fake Data'
# LINE_PNG_DIRECTORY = r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\Lines-60x15x5x0.7'
# LINE_PNG_DIRECTORY = r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\Lines-30x5x3x0.7'
# LINE_PNG_DIRECTORY = r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\Lines-75x15x5x0.7'
# LINE_PNG_DIRECTORY = r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\Lines-20x5x2x0.7'
LINE_PNG_DIRECTORY = r'C:\Users\Janet\OneDrive\Documents\SEC Quant\Triangle CNN\Lines-30x10x1x0.9'
SYMBOL_FOLDER = r'C:\Users\Janet\OneDrive\Documents\SEC Quant\Triangle CNN\Market Database'
#SYMBOL_FOLDER = r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\Market Database'
# CANDLESTICKS_32x4 = r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\32x4 Results\Triangle Candlesticks'


'''
Def:
    Function to generate candlestick plot without gridlines or axes
Params:
    ticker = company in the database
    start = start date of the plot
    end = end date of the plot
Return:
    Creates a PNG of the candlestick plot in the Candlesticks folder
'''
def make_candlestick(ticker, start_date, end_date, filepath):
    # Format string date
    start_date = str(datetime.datetime.strptime(start_date, "%Y-%m-%d"))[:10]
    end_date = str(datetime.datetime.strptime(end_date, "%Y-%m-%d"))[:10]
    # Get the CSV file containing the ticker information
    df = pd.read_csv(FILE_DIRECTORY + '\\' + str(ticker)+".csv")
    # Convert date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    # Get only the rows that are in between the start and end date, inclusive
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    df.reset_index
    df.head()
    # Get only the columns we need (date, open, close, high, low)
    df = pd.DataFrame(df, columns=["Date","Open","Close","High","Low"])
    # Differentiate between increasing and decreasing
    inc = df.Close > df.Open
    dec = df.Open > df.Close
    w = 12*60*60*1000 # half day in ms
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    # Graphing
    p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, title = str(ticker) + " Candlestick")
    p.xaxis.major_label_orientation = pi/4
    p.grid.grid_line_alpha=0.3
    p.xaxis.major_label_overrides = {i: date.strftime('%b %d') for i, date in enumerate(pd.to_datetime(df["Date"]))}
    # use the *indices* for x-axis coordinates, overrides will print better labels
    # avoids the issue of having time series data with gaps
    p.segment(df.index, df.High, df.index, df.Low, color="black")
    p.vbar(df.index[inc], 0.5, df.Open[inc], df.Close[inc], fill_color="#D5E1DD", line_color="black")
    p.vbar(df.index[dec], 0.5, df.Open[dec], df.Close[dec], fill_color="#F2583E", line_color="black")
    # Export as PNG
    export_png(p, filepath +'\\'+str(ticker)+' '+str(start_date)+' '+str(end_date)+" Candlestick.png")


'''
Def:
    Function that calculates the slope and y-intercept of the best fit line
Params:
    x = list of x values
    y = list of y values
Return:
    m = slope
    b = y-intercept
'''
def best_fit_line(x, y):
    # calculate the slope given arrays for x and y
    m = (((np.mean(x)*np.mean(y)) - np.mean(x*y)) / ((np.mean(x)*np.mean(x)) - np.mean(x*x)))
    # calculate y-intercept
    b = np.mean(y) - m*np.mean(x)
    return m, b


'''
Def:
    Calculates SSE of the regression line vs the original line
Params:
    ys_orig = original line
    ys_line = created regression line
Return:
    SSE = sum of squared errors
'''
def squared_error(ys_orig, ys_line):
    SSE = sum((ys_line - ys_orig) * (ys_line - ys_orig))
    return SSE


'''
Def:
    Calculates the r^2 of the generated regression line
Params:
    ys_orig = original line
    ys_line = regression line
Return:
    r_squared = r^2 of the regression line
'''
def r_squared(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    r_squared = 1 - (squared_error_regr/squared_error_y_mean)
    return r_squared


# '''
# Def:
#     Takes the highs and lows from the csv and find "confirmation points" for n local maxima and n local minima
# Params:
#     df = dataframe containing the OHLC data
#     n = minimum days between local maxima and local minima
# Return:
#     absoluteMax = array of n local maxima
#     absoluteMax_index = array of indices for respective local maxima (for the sake of plotting later)
#     absoluteMin = array of n local minima
#     absoluteMin_index = array of indices for respective local minima
# '''
# def get_confirmation_points(df, n):
#     highs = np.array(df['High'])
#     lows = np.array(df['Low'])
#     dates = np.array(df['Date'])
#     days = highs.size
#     localMax = []; localMin = []
#     localMax_index = []; localMin_index = []
#     for x in range(1, days-1):
#         if highs[x-1] <= highs[x] >= highs[x+1]:
#             if len(localMax) == 0:
#                 localMax.append(highs[x])
#                 localMax_index.append(x)
#             else:
#                 if x >= localMax_index[-1] + n:
#                     localMax.append(highs[x])
#                     localMax_index.append(x)
#         if lows[x-1] >= lows[x] <= lows[x+1]:
#             if len(localMin) == 0:
#                 localMin.append(lows[x])
#                 localMin_index.append(x)
#             else:
#                 if x >= localMin_index[-1] + n:
#                     localMin.append(lows[x])
#                     localMin_index.append(x)
#     return localMax, localMax_index, localMin, localMin_index


'''
Def:
    Takes the highs and lows from the csv and find "confirmation points" for n local maxima and n local minima
Params:
    df = dataframe containing the OHLC data
    n = number local maxima an dminima
Return:
    absoluteMax = array of n local maxima
    absoluteMax_index = array of indices for respective local maxima (for the sake of plotting later)
    absoluteMin = array of n local minima
    absoluteMin_index = array of indices for respective local minima
'''
def get_confirmation_points(df, n):
    highs = np.array(df['High'])
    lows = np.array(df['Low'])
    dates = np.array(df['Date'])
    days = highs.size
    localMax = []; localMin = []
    absoluteMax = []; absoluteMax_index = []
    absoluteMin = []; absoluteMin_index = []
    for x in range(1, days-1):
        # checks if the point is a local max
        if highs[x-1] <= highs[x] >= highs[x+1]:
            localMax.append(highs[x])
        # checks if the point is a local min
        if lows[x-1] >= lows[x] <= lows[x+1]:
            localMin.append(lows[x])
    # finds the 3 largest local maxima and 3 lowest minima
    # appends to the absoluteMax and absoluteMin arrays respectively 
    for i in range(n):
        max_element = max(localMax)
        absoluteMax.append(max_element)
        absoluteMax_index.append(localMax.index(max_element))
        localMax.remove(max_element)
        min_element = min(localMin)
        absoluteMin.append(min_element)
        absoluteMin_index.append(localMin.index(min_element))
        localMin.remove(min_element)
    return absoluteMax, absoluteMax_index, absoluteMin, absoluteMin_index



'''
Def:
    Function that plots the regression line of the confirmation points
        Also prints out the number of successfully generated plots
Params:
    ticker = stock's ticker
    df = sliced dataframe
    correlation_cutoff = r^2 threshold needed in order for the regression to be plotted
    start = start date
    end = end date
'''
successful_plots = 0 # global variable for all successful plots
def make_lines(ticker, df, correlation_cutoff, start, end):
    df = pd.DataFrame(df, columns=["Date","High","Low"])
    df.reset_index
    global successful_plots
    # gets the confirmation points for local minima nad maxima from the dataset
    highs, high_index, lows, low_index = get_confirmation_points(df, 5)
    # x and y values for maxima
    y_high = np.array(highs, dtype=np.float64)
    x_high = np.array(high_index, dtype=np.float64)
    y_low = np.array(lows, dtype=np.float64)
    # x and y values for minima
    x_low = np.array(low_index, dtype=np.float64)
    # generates slope and intercept for maxima and minima regression lines
    m_high, b_high = best_fit_line(x_high, y_high)
    m_low, b_low = best_fit_line(x_low, y_low)
    # creates regression for maxima and minima
    high_regression_line = [((m_high*x) + b_high) for x in x_high]
    low_regression_line = [((m_low*x) + b_low) for x in x_low]
    # calculates correlation for maxima and minima regression lines
    high_rsquared = r_squared(y_high, high_regression_line)
    low_rsquared = r_squared(y_low, low_regression_line)
    # plots both lines only if both satisfy a correlation coefficient threshold
    if high_rsquared>=correlation_cutoff and low_rsquared>=correlation_cutoff:
        plt.plot(x_high, high_regression_line, color='black')
        plt.plot(x_low, low_regression_line, color='black')
        successful_plots += 1
        plt.axis('off')
        name = str(ticker)+' '+str(start)+' '+str(end)+' ' + 'Line.png'
        plt.savefig(LINE_PNG_DIRECTORY+"\\"+str(name))
        plt.cla()
        # clear_output()
        print(successful_plots)


'''
Def:
    Function that creates the training dataset of lineplots for the CNN
Params:
    triangle_cutoff = how many triangles we want for training
    non_triangle_cutoff = how many non-triangles we want for training
'''
def make_training_data(triangle_cutoff, non_triangle_cutoff):
    global train_triangle
    global train_not_triangle
    # temp file paths for testing code
    triangle_path = r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\Fake Data\Triangle\\'
    not_triangle_path = r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\Fake Data\Not Triangle\\'

    x1 = random.randint(0, 10)
    x2 = random.randint(x1+1, 11)
    x3 = random.randint(0,10)
    x4 = random.randint(x3+1, 11)
    y_high = random.randint(5, 11)
    high_diff = random.randint(-10, 11)
    y_low = random.randint(0, 6)
    low_diff = random.randint(-10, 11)

    high1 = y_high
    high2 = high1 + high_diff
    low1 = y_low
    low2 = low1 + low_diff

    if low2>0 and high2>0 and high2 >= low2 and high1 > low1:
        # condition for creating a "dummy" triangle
        # only want triangles that have opposing slopes, or one line is flat and the other line is converging towards the flat line
        if (high1==high2 and low2>low1+2) or (low1==low2 and high1>high2+2) or (high1>high2+1 and low2>low1+1):
            if train_triangle < triangle_cutoff and (abs(x2-x4)<=2):
                plt.plot([x1,x2], [high1, high2], color='black')
                plt.plot([x3,x4], [low1, low2], color='black')
                plt.axis('off')
                train_triangle += 1
                plt.savefig(triangle_path+'Triangle '+str(train_triangle)+'.png')
                plt.cla()
                print('Triangles:', train_triangle)
        else:
            if train_not_triangle < non_triangle_cutoff:
                plt.plot([x1,x2], [high1, high2], color='black')
                plt.plot([x3,x4], [low1, low2], color='black')
                plt.axis('off')
                train_not_triangle += 1
                plt.savefig(not_triangle_path+'Not Triangle '+str(train_not_triangle)+'.png')
                plt.cla()


'''
DOCUMENTATION
'''
def accumulate_training_data(DATA_DIR, categories):
    training_data = []
    img_width = int(640/4) # Change image width and height to save memory
    img_height = int(480/4)
    for category in categories:
        path = os.path.join(DATA_DIR, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_width, img_height))
                training_data.append([new_array, class_num])
            except:
                pass
    return training_data


'''
DOCUMENTATION
'''
def preprocess_training_data():
    training_data = accumulate_training_data(DATA_DIR, ["Triangle", "Not Triangle"])
    random.shuffle(training_data)
    img_width = int(640/4)
    img_height = int(480/4)
    X2 = []
    y2 = []
    for features, label in training_data:
        X2.append(features)
        y2.append(label)
    X2 = np.array(X2).reshape(-1, img_width, img_height, 1)
    pickle_out = open("X2.pickle", "wb")
    pickle.dump(X2, pickle_out)
    pickle_out.close()
    pickle_out = open("y2.pickle", "wb")
    pickle.dump(y2, pickle_out)
    pickle_out.close()


'''
Def:
    Function that slices the csv for a ticker by frame size and increment to create an appropirate df
    Plots the regression lines for highs and lows for each sliced df
Params:
    ticker = stock's ticker
    frame_size = number of trading days contained within the slice
    increment = number of trading days the slice is shifted after each iteration
'''
def slice(ticker, frame_size, increment):
    df = pd.read_csv(SYMBOL_FOLDER + '\\' + str(ticker) + '.csv')
    # slices df based on frame size and incremen
    for i in range(0, len(df.index)-frame_size-1, increment):
        temp_df = df[i:i+frame_size]
        start = temp_df.at[i, 'Date']
        end = temp_df.at[i+frame_size-1, 'Date']
        # generates lines using the sliced dataframe
        make_lines(ticker, temp_df, 0.8, start, end)


'''
Def:
    Itereates through each csv contained within the market database
    Slices each csv based on frame_size and inrement
Params:
    frame_size = number of trading days contained within each slice
    increment = number of trading days the slice is shifted after each iteration
'''
def make_slice_lines(frame_size, increment):
    path = r'C:\Users\Janet\OneDrive\Documents\SEC Quant\Triangle CNN\Market Database\*csv'
    #path = r"C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\Market Database\*csv"
    for fname in glob.glob(path):
        fname_list = fname.split("\\")
        ticker = fname_list[-1][:-4]
        try:
            slice(ticker, frame_size, increment)
        except Exception as e:
            print(e)


'''
Def:
    Prepares the images before being fed into the CNN
Params:
    filepath = path that contains the images to be fed into CNN
Return:
    adjusted numpy array which can be handled by the CNN
'''
def prepare(filepath):
    img_width = int(640/4) # same as when loading data
    img_height = int(480/4)
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_width, img_height))
    return new_array.reshape(-1, img_width, img_height, 1)


'''
Def:
    Function that generates the corresponding candlestick plot whenever th CNN detects a triangle
Params:
    filepath = path containing images to be fed into CNN
    folder = folder where the generated candlesticks are stored
'''
def gen_corresponding_candlestick(filepath, folder):
    buffer = filepath.split('\\')[-1].split()
    ticker = buffer[0]; start = buffer[1]; end = buffer[2]
    make_candlestick(ticker, start, end, folder)


'''
INSERT DOCUMENTATION
'''
def train_and_save_model(c_layers, nodes, d_layers):
        X2 = pickle.load(open("X2.pickle", "rb"))
        y2 = pickle.load(open("y2.pickle", "rb"))
        X2 = X2 / 255.0 

        # 3-conv-128-node-2-dense-CNN
        conv_layers = [c_layers]
        layer_sizes = [nodes] 
        dense_layers = [d_layers]
        for dense_layer in dense_layers:
            for layer_size in layer_sizes:
                for conv_layer in conv_layers:
                    name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
                    # tensorboard = TensorBoard(log_dir='logtest/{}'.format(name))
                    print('\n'+name+'\n')
                    model = Sequential()

                    model.add(Conv2D(layer_size, (3,3), input_shape = X2.shape[1:]))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2,2)))
                    
                    for l in range(conv_layer-1):
                        model.add(Conv2D(layer_size, (3,3)))
                        model.add(Activation('relu'))
                        model.add(MaxPooling2D(pool_size=(2,2)))
                    
                    model.add(Flatten()) # converts to 1D feature vectors
                    for l in range(dense_layer): 
                        model.add(Dense(layer_size))
                        model.add(Activation('relu'))

                    model.add(Dense(1))
                    model.add(Activation('sigmoid'))
                    model.compile(loss='binary_crossentropy', 
                                optimizer='adam',
                                metrics=['accuracy'])

                    model.fit(X2, y2, batch_size=32, epochs = 20, validation_split=0.4)
        model.save('{}-conv-{}-node-{}-dense-CNN.model'.format(c_layers, nodes, d_layers))

if __name__ == "__main__":
    # # Global Variables
    # train_triangle = 0
    # train_not_triangle = 0
    # triangle_cutoff = 10000
    # non_triangle_cutoff = 10000

    # # Generate Training Data
    # while train_triangle < triangle_cutoff or train_not_triangle < non_triangle_cutoff:
    #     make_training_data(triangle_cutoff, non_triangle_cutoff)
    # preprocess_training_data()

    make_slice_lines(60, 15)

    # # Train Models 
    # train_and_save_model(3, 128, 2)
    # train_and_save_model(4, 128, 2)
    # train_and_save_model(4, 128, 4)


    # # GENERATING FINAL IMAGES USING TRAINED MODELS
    # # Uses each CNN model to evaluate the dataset generated from S&P 500 data
    # # Test the CNN models with actual data

    # triangles = 0
    # path = r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\Lines-30x5x3x0.7\*png'
    # categories = ["Triangle", "Not Triangle"]
    # models = [r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\3-conv-128-node-2-dense-CNN.model', 
    #           r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\4-conv-128-node-2-dense-CNN.model', 
    #           r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\4-conv-128-node-4-dense-CNN.model']
    # line_results_files = [r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\3-conv-128-node-2-dense-results\Triangle Lines',
    #                       r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\4-conv-128-node-2-dense-results\Triangle Lines',
    #                       r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\4-conv-128-node-4-dense-results\Triangle Lines']
    # candlestick_results_files = [r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\3-conv-128-node-2-dense-results\Triangle Candlesticks',
    #                              r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\4-conv-128-node-2-dense-results\Triangle Candlesticks',
    #                              r'C:\Users\Kevin Duan\OneDrive\Documents\SEC Quant\Triangle CNN\4-conv-128-node-4-dense-results\Triangle Candlesticks']

    # for i in range(len(models)):
    #     model = tf.keras.models.load_model(models[i])
    #     triangles = 0
    #     for fname in glob.glob(path):
    #         prediction = model.predict([prepare(fname)])
    #         if categories[int(prediction[0][0])] == "Not Triangle":
    #             triangles += 1
    #             shutil.copy(fname, line_results_files[i])
    #             gen_corresponding_candlestick(fname, candlestick_results_files[i])
    #             print(models[i], triangles)
