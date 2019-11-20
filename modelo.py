#!/usr/bin/env python3

import tensorflow as tf
import numpy as np #arrays, algebra lineal
import glob
import os
from tensorflow.python.platform import gfile
from lib.src.facenet import get_model_filenames
from lib.src.align.detect_face import detect_face  #deteccion de caras MTCNN
from lib.src.facenet import load_img
from scipy.misc import imresize, imsave
from collections import defaultdict
from flask import flash #servidor de back


def allowed_file2(filename, allowed_set):
    """Comprueba si la extension del nombre del archivo, es una extension permitida para cargar
    Parametros:
    filename: nombre del archivo cargado a verificar.
    allowed_set: conjuto que contiene las extensiones de archivo de imagenes validas.

    Retorna:
    check: valor booleano para representar si la extension de archivo esta en las permitidas
    True = el archivo esta permitido.
    False = el archivo no esta permitido. """
    check = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set
    return check


def remove_file_extension(filename):
    #Retorna el nombre del archivo sin la extension del archivo 
    filename = os.path.splitext(filename)[0]
    return filename


def save_image(img, filename, uploads_path):
    """ Guardar una imagen en la carpeta de /uploads
    Parametros:
        img: imagen en formato de array de numpy
        filename: filename of the image file.
        uploads_path: absolute path of the 'uploads/' folder.
    """
    try:
        imsave(os.path.join(uploads_path, filename), arr=np.squeeze(img))
        flash("Image saved!")
    except Exception as e:
        print(str(e))
        return str(e)


def load_model(model):
    """ Cargar el modelo de Facenet desde la ruta.

    Parametros:
        model: direccion path del modelo

    Retorna:
        graph: modelo de Tensorflow
    """

    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Nombre del Modelo de Facenet: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name='')
            return graph
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        graph = saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
        return graph

#Este metodo solo detecta una cara humana
def get_face(img, pnet, rnet, onet, image_size):
    """
    Recorta la imagen que contengan un solo rostro humano, si existe, se utiliza un MTCNN
    para detectar la cara. Luego cambia el tamano de la imagen a 160x3x3.
    Si no se detecta ninguna cara, devuelve un valor nulo.
    Parametros:
          img: imagen en formato de numpy array
          pnet: proposal net, primera etapa de la deteccion de rostros MTCNN
          rnet: refinement net,segunda etapa de la deteccion de rostros MTCNN
          onet: output net, tercera etapa de la deteccion de rostros MTCNN
          image_size: (int) Tamano de la imagen cuadrado

    Retorna:
          face_img: una imagen que contiene una cara de tamano 160x160x3
          si no se detecta un rostro humano retorna un valor nulo.
    """
    # Constantes predeterminadas de la implementación del repositorio de FaceNet de MTCNN
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    margin = 44
    input_image_size = image_size

    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face(
        img=img, minsize=minsize, pnet=pnet, rnet=rnet,
        onet=onet, threshold=threshold, factor=factor
    )

    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            det = np.squeeze(face[0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]: bb[3], bb[0]:bb[2], :]
            face_img = imresize(arr=cropped, size=(input_image_size, input_image_size), mode='RGB')
            return face_img
    else:
        return None

#Detecta multiples caras
def get_faces_live(img, pnet, rnet, onet, image_size):
    """ Detecta multiples caras humanas

    Parametros:
          img: imagen en formato de numpy array
          pnet: proposal net, primera etapa de la deteccion de rostros MTCNN
          rnet: refinement net,segunda etapa de la deteccion de rostros MTCNN
          onet: output net, tercera etapa de la deteccion de rostros MTCNN
          image_size: (int) Tamano de la imagen cuadrado

     Retorna:
           faces: Lista que contiene los rostros humanos recortados
           rects: Lista que contiene las coordenadas del rectangulo que se dibujara alrededor del rostro.
    """
    # Constantes predeterminadas de la implementación del repositorio de FaceNet de MTCNN
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    margin = 44
    input_image_size = image_size

    faces = []
    rects = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face(
        img=img, minsize=minsize, pnet=pnet, rnet=rnet,
        onet=onet, threshold=threshold, factor=factor
    )
    # Si una o varias imagenes humanas son detectadas
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = imresize(arr=cropped, size=(input_image_size, input_image_size), mode='RGB')
                faces.append(resized)
                rects.append([bb[0], bb[1], bb[2], bb[3]])

    return faces, rects


def forward_pass(img, session, images_placeholder, phase_train_placeholder, embeddings, image_size):
    """Ingresa una imagen al modelo de FaceNet y devuelve un embedding de 128 dimensiones para el reconocimiento facial
    Parametros:
        img: imagen en formato de array de numpy 
        session: Activa una session en Tensorflow 
        images_placeholder: placeholder del 'input:0' tensor del modelo preentrenado de FaceNet
        phase_train_placeholder: placeholder of the 'phase_train:0' tensor of the pre-trained FaceNet model graph.
        embeddings: placeholder of the 'embeddings:0' tensor from the pre-trained FaceNet model graph.
        image_size: (int) required square image size.

    Retorna:
          embedding: Vector de 128 dimensiones despues de que la imagen pasa por el modelo de Facenet
          si no detecta nada, regresa un None
    """
    # Si hay una cara en la imagen 
    if img is not None:
        #Normalizar los valores de pixeles de la imagen para reducir el ruido y mejorar la presicion
        image = load_img(
            img=img, do_random_crop=False, do_random_flip=False,
            do_prewhiten=True, image_size=image_size
        )
        # Ejecuta el forward pass en el modelo de FaceNet para calcular el embedding
        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
        embedding = session.run(embeddings, feed_dict=feed_dict)
        return embedding

    else:
        return None


def save_embedding(embedding, filename, embeddings_path):
    """ Guardar el embedding en formato numpy en la carpeta embeddings
    Parametros:
        embedding: Vector de numpy de 128 dimensiones despues de que paso por el modelo de FaceNet
        filename: nombre de la imagen del input que ingresa el usuario
        embeddings_path: absolute path de la carpeta 'embeddings/'
    """
    #Guardar el embedding usando el nombre de la imagen
    #esto se modifico y se pasa el input del frontend
    path = os.path.join(embeddings_path, str(filename))
    try:
        np.save(path, embedding)

    except Exception as e:
        print(str(e))


def load_embeddings():
    """
        Carga los embeddings de numy en la carpeta de embedding en un objeto  defaultdict
    Retorna:
        embedding_dict: defaultdict contiene el embedding numpy y el nombre del embedding
    """
    embedding_dict = defaultdict()

    for embedding in glob.iglob(pathname='embeddings/*.npy'):
        name = remove_file_extension(embedding)
        dict_embedding = np.load(embedding)
        embedding_dict[name] = dict_embedding

    return embedding_dict


def identify_face(embedding, embedding_dict):
    """
        Compara el embedding recibido  con los embeddings de la carpeta embeddings.
        La distancia euclidiana minima (o norma del vector), el embedding con la menor distancia es la clase predicha.

    Si todas las incrustaciones tienen una distancia por encia del threshold  1.1 entonces la imagen no existe en la carpeta de embeddings

    Parametros:
        embedding: Array de Numpy que contiene el embedding que se comparara con las incrustaciones de la distancia eucliana.
        embedding_dict: (defaultdict)  contiene el nombre del archivo del embedding y del array del numpy

    Retorna:
          result: (string) Describe la persona que sea mas probable que coincida con la cara
        si no existe en la base de datos, es que la distancia esta por encima del threshold  
    """
    min_distance = 100
    try:
        for (name, dict_embedding) in embedding_dict.items():
            # Calcular la distancia euclidiana entre el embedding actual y los embeddings de la carpeta 'embeddings
            distance = np.linalg.norm(embedding - dict_embedding)

            if distance < min_distance:
                min_distance = distance
                identity = name

        if min_distance <= 1.1:
            # remove 'embeddings/' from identity
            identity = identity[11:]
            result = " Es " + str(identity) + ", la distancia es " + str(min_distance)
            return result

        else:
            result = "Desconocido, la distancia es " + str(min_distance)
            return result

    except Exception as e:
        print(str(e))
        return str(e)
