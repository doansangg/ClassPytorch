import os
train=open("path_data/test.txt","w")
config=open("path_data/config_test.txt","w")
path_image="/media/sang/UBUNTU/DATA"
folder=sorted(os.listdir(path_image))
for count,fd in enumerate(folder):
  label= str(count) +'\t'+ fd +"\n"
  config.write(label)
  for sang,i in enumerate(os.listdir(path_image+'/'+fd)):
    if (len(i.split('(1)'))<2):
      image=path_image+'/'+fd+'/'+i
      string=image+"\t"+str(count)+"\n"
      if os.path.exists(image):
        train.write(string)
    
        
          