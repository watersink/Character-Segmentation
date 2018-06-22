import cv2
import numpy as np
import os
import random
from PIL import Image,ImageDraw,ImageFont


class Images_make(object):
    def __init__(self):
        self.image_width=2048
        self.image_height=64
        self.character_width = 64
        self.character_height = 64
        self.character_channel = 3

        self.max_character_num=26
        self.max_character_space=8
        self.max_character_before=self.character_width*2

        self.character_type=( './font_style/simfang.ttf','./font_style/simhei.ttf','./font_style/simkai.ttf' ,'./font_style/simsun.ttc','./font_style/Songti_SC_Regular.ttf','./font_style/STKAITI.TTF','./font_style/STSONG.TTF')
        self.img_path="./DATA/IMAGES/"
        self.label_path="./DATA/LABELS/"

        self.character_dictionary = {}
        self.character_file = 'tag.txt'
        #generate how many images
        self.generate_num=5

        self.draw_rectangle=False

    def generate_character_line(self):
        num=0

        for i in range(self.generate_num):
            character_nums = random.randint(1, self.max_character_num)
            character_space = random.randint(self.max_character_space - 4, self.max_character_space + 4)
            character_before = random.randint(0, self.max_character_before)
            character_type = self.character_type[random.randint(0, 6)]

            font = ImageFont.truetype(character_type, self.character_height)
            image = Image.new('RGB', (self.image_width, self.image_height), (255, 255, 255))
            x1 = character_before
            y1 = 0
            x2 = x1 + self.character_width
            y2 = self.character_height

            f=open(os.path.join(self.label_path,str(num).zfill(5)+'.txt'),'w')
            for label_f in range(x1):
                f.write(str(0)+" ")

            for character_num in range(character_nums):
                char_num = random.randint(0, len(self.character_dictionary) - 1)
                image_roi = image.crop((x1, y1, x2, y2))
                draw = ImageDraw.Draw(image_roi)
                if character_type == 'simfang.ttf' or character_type == 'simhei.ttf' or character_type == 'simkai.ttf' or character_type == 'simsun.ttc':
                    draw.text((0, 0), self.character_dictionary[str(char_num)], (0, 0, 0), font=font)
                else:
                    draw.text((0, -12), self.character_dictionary[str(char_num)], (0, 0, 0), font=font)
                image.paste(image_roi, (x1, y1, x2, y2))

                if self.draw_rectangle==True:
                    drawObject = ImageDraw.Draw(image)
                    drawObject.rectangle((x1, y1, x2, y2),fill=None,outline='red')

                for label_f in range(x1,x2):
                    f.write(str(1) + " ")
                for label_f in range(character_space):
                    f.write(str(0) + " ")

                x1 = x2 + character_space
                x2 = x1 + self.character_width

            for label_f in range(x1,self.image_width):
                f.write(str(0) + " ")
            f.write("\n")
            f.close()
            image.save(self.img_path + str(num).zfill(5) + '.jpg')
            num += 1
            del image
            print("processed %d pic"%(i))


    def load_dictionary(self):
        for line in open(self.character_file, 'r', encoding='gbk'):
            classnum, character = line.split(" ")
            assert (classnum is not None)
            assert (character is not None)
            self.character_dictionary[str(classnum)] = str(character)
        return self.character_dictionary




if __name__=="__main__":
    images_make = Images_make()
    images_make.load_dictionary()
    images_make.generate_character_line()