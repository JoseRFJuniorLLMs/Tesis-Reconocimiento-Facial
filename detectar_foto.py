from preprocess import preprocesses

input_datadir = './train_img'
output_datadir = './pre_img'

obj=preprocesses(input_datadir,output_datadir)
nrof_images_total,nrof_successfully_aligned=obj.collect_data()

print('El numero total de imagenes es: %d' % nrof_images_total)
print('Numero total de imagenes alineadas: %d' % nrof_successfully_aligned)



