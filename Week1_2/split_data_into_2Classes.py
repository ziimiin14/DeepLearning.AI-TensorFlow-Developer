import os
from shutil import copyfile

base_path = 'dog_cat_dataset'

train_path = os.path.join(base_path,'train')

train_list = os.listdir(train_path)

dog = []
cat = []

for x in train_list:
    if x[0:3] == 'dog':
        dog.append(x)
    else:
        cat.append(x)

count = 0
des_x = 'dog_cat_classes/dog'
des_y = 'dog_cat_classes/cat'


for x,y in zip(dog,cat):
    temp_x = os.path.join(train_path,x)
    temp_y = os.path.join(train_path,y)
    temp_x1 = os.path.join(des_x,str(count)+'.jpg')
    temp_y1 = os.path.join(des_y,str(count)+'.jpg')
    copyfile(temp_x,temp_x1)
    copyfile(temp_y,temp_y1)
    count += 1

print('done')





