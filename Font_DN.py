# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import os
import os.path
import random
import numpy as np


class Font:
    
    def __init__(self, size, root_dir, input_letter, output_letter):
        self.size = size
        self.input_letter = input_letter
        self.output_letter = output_letter
        
        font_files = []
        for parent,dirnames,filenames in os.walk(root_dir):  
            for filename in filenames:
                if filename[-3:] == 'ttf' or filename[-3:] == 'TTF':
                    font_files.append(os.path.join(parent,filename))
        print(('Fond %i font files') % (len(font_files)))
        random.shuffle(font_files)
        self.font_files = font_files


    def getSize(self):
        return(self.size)

    def getLetterSets2(self,n_test_examples,output):
        test_input1=np.zeros((n_test_examples,1,self.size,self.size))
        wi = int(self.size * 0.9)
        le = int(self.size * 0.05)
        d_font=ImageFont.truetype("Fonts/simsun.ttf",wi)
        letter=output[0]
        for m in range(0,n_test_examples):
            img=Image.new('L',(self.size,self.size),0)
            draw=ImageDraw.Draw(img)
            draw.text((le,le),letter,1,font=d_font)
            draw=ImageDraw.Draw(img)
            test_input1[m,0,:]=np.array(img);
        return test_input1

    def getLetterSets1(self,n_train_examples,output):
        train_input1=np.zeros((n_train_examples,1,self.size,self.size))
        wi = int(self.size * 0.9)
        le = int(self.size * 0.05)
        d_font=ImageFont.truetype("Fonts/simsun.ttf",wi)
        letter=output[0]
        for m in range(0,n_train_examples):
            img=Image.new('L',(self.size,self.size),0)
            draw=ImageDraw.Draw(img)
            draw.text((le,le),letter,1,font=d_font)
            draw=ImageDraw.Draw(img)
            train_input1[m,0,:]=np.array(img);
        return train_input1




        
    def getLetterSets(self, n_train_examples, n_test_examples):
        # return a 4D numpy array that contains images of multiple letters
        train_input = np.zeros((n_train_examples, len(self.input_letter)+1,self.size,self.size))
        train_output = np.zeros((n_train_examples))
        test_input = np.zeros((n_test_examples, len(self.input_letter)+1,self.size,self.size))
        test_output = np.zeros((n_test_examples))
        wi = int(self.size * 0.9)
        le = int(self.size * 0.05)
        n_same=0
        #training data, 5*36*36 pictures->true/false
        for i in range(0,n_train_examples):
            if(n_same>int(n_train_examples/2)):
                is_same=False
            else:
                is_same=random.choice([True,False])
            base_font=random.choice(self.font_files)
            if(is_same):
                out_font=base_font
                n_same+=1
            else:
                out_font=random.choice(self.font_files)
                while(out_font==base_font):
                    out_font=random.choice(self.font_files)
            n=0
            #generate BASQ
            for letter in self.input_letter:
                font = ImageFont.truetype(base_font, wi)
                img = Image.new('L',(self.size,self.size),0)      # orignial: L and (1)
                draw = ImageDraw.Draw(img)
                draw.text((le, le),letter,1,font = font)      # original: (0)
                draw = ImageDraw.Draw(img)
                train_input[i, n, :, :] = np.array(img)
                n = n + 1
            #generate R
            for letter in self.output_letter:
                font = ImageFont.truetype(out_font, self.size)
                img = Image.new('L',(self.size,self.size),(1))
                draw = ImageDraw.Draw(img)
                draw.text((le, le),letter,(0),font = font)
                draw = ImageDraw.Draw(img)
                train_input[i, n, :, :] = np.array(img)
            #generate final boolean result, 1 for same, 0 for different
            if(is_same):
                train_output[i]=1;
            else:
                train_output[i]=0;
        n_same=0

        for i in range(0,n_test_examples):
        #testing data, the same format
            if(n_same>int(n_test_examples/2)):
                is_same=False
            else:
                is_same=random.choice([True,False])
            base_font=random.choice(self.font_files)
            if(is_same):
                out_font=base_font
                n_same+=1
            else:
                out_font=random.choice(self.font_files)
                while(out_font==base_font):
                    out_font=random.choice(self.font_files)
            n=0
            #generate BASQ
            for letter in self.input_letter:
                font = ImageFont.truetype(base_font, wi)
                img = Image.new('L',(self.size,self.size),0)      # orignial: L and (1)
                draw = ImageDraw.Draw(img)
                draw.text((le, le),letter,1,font = font)      # original: (0)
                draw = ImageDraw.Draw(img)
                test_input[i, n, :, :] = np.array(img)
                n = n + 1
            #generate R
            for letter in self.output_letter:
                font = ImageFont.truetype(out_font, self.size)
                img = Image.new('L',(self.size,self.size),(1))
                draw = ImageDraw.Draw(img)
                draw.text((le, le),letter,(0),font = font)
                draw = ImageDraw.Draw(img)
                test_input[i, n, :, :] = np.array(img)
            #generate final boolean result, 1 for same, 0 for different
            if(is_same):
                test_output[i]=1;
            else:
                test_output[i]=0;
        return (train_input, train_output, test_input, test_output)