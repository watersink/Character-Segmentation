
import cv2
import numpy as np
import os
import random
import time
from PIL import Image,ImageDraw,ImageFont

from data_util import GeneratorEnqueuer



def load_dictionary(character_file = './tag.txt'):
    character_dictionary = {}


    for line in open(character_file, 'r', encoding='gbk'):
        classnum, character = line.split(" ")
        assert (classnum is not None)
        assert (character is not None)
        character_dictionary[str(classnum)] = str(character)
    return character_dictionary

def generator_on_the_fly(image_width=2048,image_height=64,image_channel=3, batch_size=10,display=False):
        
    character_width = 64
    character_height = 64
    character_channel = image_channel

    max_character_num=26
    max_character_space=8
    max_character_before=character_width*2
    character_type_tuple=('./font_style/simfang.ttf','./font_style/simhei.ttf','./font_style/simkai.ttf' ,'./font_style/simsun.ttc','./font_style/Songti_SC_Regular.ttf','./font_style/STKAITI.TTF','./font_style/STSONG.TTF')

    character_dictionary=load_dictionary()


    while True:
        try:    
            gene_images=np.zeros((batch_size,image_height,image_width,image_channel),np.float32)
            gene_labels=np.zeros((batch_size,image_width),np.float32)

            for i in range(batch_size):
                character_nums = random.randint(1, max_character_num)
                character_space = random.randint(max_character_space - 4, max_character_space + 4)
                character_before = random.randint(0, max_character_before)
                character_type = character_type_tuple[random.randint(0, 6)]

                font = ImageFont.truetype(character_type, character_height)
                image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
                x1 = character_before
                y1 = 0
                x2 = x1 + character_width
                y2 = character_height


                for character_num in range(character_nums):
                    char_num = random.randint(0, len(character_dictionary) - 1)
                    image_roi = image.crop((x1, y1, x2, y2))
                    draw = ImageDraw.Draw(image_roi)
                    if character_type == 'simfang.ttf' or character_type == 'simhei.ttf' or character_type == 'simkai.ttf' or character_type == 'simsun.ttc':
                        draw.text((0, 0), character_dictionary[str(char_num)], (0, 0, 0), font=font)
                    else:
                        draw.text((0, -12), character_dictionary[str(char_num)], (0, 0, 0), font=font)
                    image.paste(image_roi, (x1, y1, x2, y2))

                    gene_labels[i,x1:x2]=1

                    if display==True:
                        drawObject = ImageDraw.Draw(image)
                        drawObject.rectangle((x1, y1, x2, y2),fill=None,outline='red')

                    x1 = x2 + character_space
                    x2 = x1 + character_width
        
                if display==True:
                    print("labels:",gene_labels[i,:])
                    cv2.imshow("img",np.asarray(image))
                    cv2.waitKey()

                gene_images[i,:,:,:]=image
                del image

            gene_images=gene_images*1.0/255
            yield gene_images,gene_labels
        except Exception as e:
                import traceback
                traceback.print_exc()
                continue
    



def generator_from_folder(image_width=2048,image_height=64,image_channel=3, batch_size=10):
    pics_list = []
    labels_list = []
    img_dirs = './DATA/IMAGES/'
    label_dirs='./DATA/LABELS/'

    imgs_names=os.listdir(img_dirs)
    labels_names=os.listdir(label_dirs)

    while True:
        try:    
            gene_images=np.zeros((batch_size,image_height,image_width,image_channel),np.float32)
            gene_labels=np.zeros((batch_size,image_width),np.float32)

            m = random.randint(0, len(imgs_names)-batch_size-1)
            for i in range(batch_size):
                img = cv2.imread(os.path.join(img_dirs,imgs_names[m+i]))
                img = cv2.resize(img, (image_width,image_height))
                img=1.0*img/255

                f=open(os.path.join(label_dirs,labels_names[m+i]))
                line=f.read()
                line=line.strip('\n')
                line_label=line.split(' ')[:image_width]
                line_label=np.asarray(line_label,np.float32)
                f.close()

                gene_images[i,:,:,:]=img
                gene_labels[i,:]=line_label

            yield gene_images,gene_labels
        except Exception as e:
                import traceback
                traceback.print_exc()
                continue

def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator_on_the_fly(**kwargs), use_multiprocessing=False)
        #enqueuer = GeneratorEnqueuer(generator_from_folder(**kwargs), use_multiprocessing=False)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=100, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()
