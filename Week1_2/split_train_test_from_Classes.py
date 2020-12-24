import os
import random
from shutil import copyfile

def split_data(src,training,testing,split_size):
    img = os.listdir(src)
    random.sample(img,len(img))
    random.sample(img,len(img))
    random.sample(img,len(img))

    train_img =img[0:int(split_size*len(img))]
    test_img = img[int(split_size * len(img)):]


    for x in train_img:
        temp = os.path.join(src,x)
        temp1 = os.path.join(training,x)
        copyfile(temp,temp1)

    for y in test_img:
        temp = os.path.join(src, y)
        temp1 = os.path.join(testing, y)
        copyfile(temp, temp1)





src_cat = 'dog_cat_classes/cat'
src_dog = 'dog_cat_classes/dog'

training_cat_path = 'dog_cat_train_test/training/cat'
training_dog_path = 'dog_cat_train_test/training/dog'

testing_cat_path = 'dog_cat_train_test/testing/cat'
testing_dog_path = 'dog_cat_train_test/testing/dog'

split_size =.9

split_data(src_cat,training_cat_path,testing_cat_path,split_size)
split_data(src_dog,training_dog_path,testing_dog_path,split_size)


