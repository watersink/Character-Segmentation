# Character Segmentation
Segment characters and spaces in one text line,from this paper [Chinese English mixed Character Segmentation as Semantic Segmentation](https://arxiv.org/pdf/1611.01982.pdf)

## dependencies
tensorflow1.3,or 1.4
python3 

## architecture of the network
![image]( ./other/1.png)

## Heuristic Rules for balanced_Binary_CrossEntropy
![image]( ./other/2.png)

## make training images and labels
    python3 make_train_images.py

## train
    python3 train_char_seg.py
![image]( ./other/3.png)
## test
    python3 test_char_seg.py
![image]( ./other/4.png)

## other_things
you can choose first make traing images and then use these maked images to train ,or training and making at the same time.all you need to do is change below codes in data_generator.py
    enqueuer = GeneratorEnqueuer(generator_on_the_fly(**kwargs), use_multiprocessing=False)
    #enqueuer = GeneratorEnqueuer(generator_from_folder(**kwargs), use_multiprocessing=False)