import numpy as np
from SegmentPage import segment_into_lines
from SegmentLine import segment_into_words
from RecognizeWord import recognize_words




#Open image and segment into lines
line_img_array=segment_into_lines('./test_image.jpg')


#Creating lists to store the line indexes,words list.
full_index_indicator=[]
all_words_list=[]
#Variable to count the total no of lines in page.
len_line_arr=0

#Segment the lines into words and store as arrays.
for idx,im in enumerate(line_img_array):
    line_indicator,word_array=segment_into_words(im,idx)
    for k in range(len(word_array)):
        full_index_indicator.append(line_indicator[k])
        all_words_list.append(word_array[k])
    len_line_arr+=1
    

all_words_list=np.array(all_words_list)


#Perform the recognition on list of list of words.
recognize_words(full_index_indicator,all_words_list,len_line_arr)

