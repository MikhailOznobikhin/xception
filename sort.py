from os import listdir, rename
from os.path import isfile, join

source_dir = 'data/train_raw'
train_dir = 'data/train'
validation_dir = './data/validation'
test_dir = './data/test'

cats_dir = 'cats'
dogs_dir = 'dogs'

images = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]
dogs = [i for i in images if i.startswith('dog')]
print('dogs: '+ str(len(dogs)))
cats = [i for i in images if i.startswith('cat')]
print('cats: '+ str(len(cats)))

for cat in cats[:2000]:
    rename(join(source_dir, cat), join(train_dir, cats_dir, cat))
    
for cat in cats [2000:3000]:
    rename(join(source_dir, cat), join(validation_dir, cats_dir, cat))
    
for cat in cats [3000:6000]:
    rename(join(source_dir, cat), join(test_dir, cats_dir, cat))

for dog in dogs[:2000]:
    rename(join(source_dir, dog), join(train_dir, dogs_dir, dog))
    
for dog in dogs [2000:3000]:
    rename(join(source_dir, dog), join(validation_dir, dogs_dir, dog))
    
for dog in dogs [3000:6000]:
    rename(join(source_dir, dog), join(test_dir, dogs_dir, dog))

print('sort complite')
