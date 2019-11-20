from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from classifier import training

datadir = './pre_img'
modeldir = './modelo_transferlearning/20170511-185253.pb'
classifier_filename = './resultados/classifier.pkl'
print ("Entrenando")
obj=training(datadir,modeldir,classifier_filename)
get_file=obj.main_train()
print('Guardado en la carpeta resultados "%s"' % get_file)
sys.exit("Termino el entrenamiento")
