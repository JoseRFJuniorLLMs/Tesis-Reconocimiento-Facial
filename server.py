#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import sys

import datetime
import subprocess #Camara de Windows


import cv2  # este es el importe de openCV
import tensorflow as tf
import os
from scipy.misc import imread #leer imagenes
from lib.src.align import detect_face  # para la deteccion facial con MTCNN
from flask import Flask, request, render_template, sessions #backend
from werkzeug.utils import secure_filename #nombre de los archivos
from waitress import serve

#librerias para entrenamiento
from classifier import training
from preprocess import preprocesses


#librerias del modelo
from modelo import (
    load_model, get_face, get_faces_live, forward_pass, save_embedding, load_embeddings,
    identify_face, allowed_file2, remove_file_extension, save_image
)

from imagenetweb import (
    load_model2, detect_face2, display_result, face_classification, map_label

)

from face_classification import (
    load_imgs, create_labels, reshape_for_keras, train_test_split, create_model, train_and_evaluate, save_model
)


#librerias photo-gallery
from flask import redirect, url_for
from werkzeug.utils import secure_filename
import json
import os
from modules.Login import Login
from modules.Gallery import Gallery
from modules.Photos import Photos
from definition import *



#configuracion del servidor
app = Flask(__name__)
app.secret_key = os.urandom(24)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
uploads_path = os.path.join(APP_ROOT, 'uploads')
embeddings_path = os.path.join(APP_ROOT, 'embeddings')
allowed_set = set(['png', 'jpg', 'jpeg'])  # formatos de imagen a subir 


#Codigo Photo-Gallery
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg']) 
session = {}
photos_obj = Photos()

SAVE_FACE = 1
FACES_FOLDER = 'Faces'

#METODOS DE PHOTO GALERY
@app.route('/')
@app.route('/index')
def index():
    return redirect(url_for('index_page'))


@app.route('/login', methods=['POST'])
def do_login():
    #if request.method == "POST":
    #user_name = request.form['username']
    #password = request.form['password']
    user_name='admin'
    password='root'

    login_obj = Login()
    result = login_obj.login(user_name, password)
    print(result['type'])

    if result['result']:
        session['type'] = result['type']

    response = {'success': result}
    return json.dumps(response)


@app.route('/logout')
def do_logout():
    return redirect(url_for('index'))

@app.route('/galleries')
def galleries():
#if 'type' in session:
    gallery_obj = Gallery()
    galleries = gallery_obj.get_all_gallery()
    return render_template("gallery.html", galleries=galleries)
#else:



@app.route('/galleries/add', methods=['POST'])
def add_galleries():
#if 'type' in session:
    if request.method == "POST":
        gallery_name = request.form['galleryName']

        gallery_obj = Gallery()
        result = gallery_obj.add_gallery(gallery_name)

        response = {'success':result}
        return json.dumps(response)
#else:
    #response = {'success': False}
   # return json.dumps(response)


@app.route('/galleries/edit', methods=['POST'])
def edit_galleries():
    if request.method == "POST":
        new_name = request.form['newName']
        gallery_name = request.form['galleryName']

        gallery_obj = Gallery()
        result = gallery_obj.edit_gallery_name(gallery_name, new_name)

        response = {'success': result}
        return json.dumps(response)


@app.route('/galleries/delete', methods=['POST'])
def delete_galleries():
    if request.method == "POST":
        gallery_name = request.form['galleryName']

        gallery_obj = Gallery()
        result = gallery_obj.delete_gallery(gallery_name)

        response = {'success': result}
        return json.dumps(response)

# -------------- Gallery Routes ----------------


# -------------- Gallery Photos Routes ----------------
@app.route('/galleries/album/<gallery_name>', methods=['GET'])
def gallery(gallery_name):
#if 'type' in session:
    photos_obj = Photos()
    photos = photos_obj.get_all_gallery_photos(gallery_name)
    return render_template("photos.html", photos=photos, gallery_folder=Gallery_Folder,gallery_name=gallery_name)
#else:
    #return redirect(url_for("index"))


@app.route('/galleries/album/photos/delete', methods=['POST'])
def delete_gallery_photo():
    if request.method == "POST":
        gallery_name = request.form['galleryName']
        photo_name = request.form['photoName']

        result = photos_obj.delete_gallery_photos(gallery_name, photo_name)

        response = {'success': result}
        return json.dumps(response)



@app.route('/galleries/album/<gallery_name>/upload', methods=['GET','POST'])
def upload_gallery_photo(gallery_name):
    app.config['UPLOAD_FOLDER'] = Gallery_Folder+gallery_name

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(url_for('gallery', gallery_name=gallery_name))
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            msg = 'No selected file'
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            for file in request.files.getlist('file'):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return redirect(url_for('gallery', gallery_name=gallery_name))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST', 'GET'])
def get_image():
    """
     Se obtine una imagen por el POST, se obtiene un embedding que posteriormente se mandara al modelo de FaceNet
        'uploads' es la carpata para las fotos 
        'embeddings' es la carpeta para guardar los embeddings
    """

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No se cargo nigun archivo"

        file = request.files['file']
        filename = file.filename

        if filename == "":
            return "No ha seleccionado una imagen"

        if file and allowed_file2(filename=filename, allowed_set=allowed_set):
            filename = secure_filename(filename=filename)
            #Leemos una imagen como un vector de numpy con RGB
            img = imread(name=file, mode='RGB')
            # Detecta y recorta una imagen de 160 x 160 que contiene un rostro
            img = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)

            # Si se detecta una cara
            if img is not None:

                embedding = forward_pass(
                    img=img, session=facenet_persistent_session,
                    images_placeholder=images_placeholder, embeddings=embeddings,
                    phase_train_placeholder=phase_train_placeholder,
                    image_size=image_size
                )
             
                # Guardar las caras de imagenes en la carpeta 'uploads/'
                save_image(img=img, filename=filename, uploads_path=uploads_path)
            
                # Eliminar la extensión de archivo
                filename = remove_file_extension(filename=filename)
                # Guardando el embedding en la carpeta 'embeddings/'
                fullname = request.form['fullname'] 
                save_embedding(embedding=embedding, filename=fullname, embeddings_path=embeddings_path)

                return render_template("embecargado.html",
                                       status="Imagen de ",
                                       status2=fullname,
                                       status3=" cargada",
                                      status4="embedding generado existosamente")

            else:
                #Si no se detecta una cara en la imagen 
                return render_template("embecargado.html",
                                       status="No se han detectado caras en la foto.")

    else:
        return "Metodo POST es requerido"


@app.route('/predecirimagen', methods=['POST', 'GET'])
def predict_image():
    """
    Obtiene una imagen a traves del POST, pasa la imagen a traves del modelo de Facenet, 
    el embedding que resulta es enviado para ser comparado en la base de datos de embeddings.
    Luego en otra pagina HTML se muestra el resultado de la prediccion
    """

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No hay archivo"

        file = request.files['file']
        filename = file.filename

        if filename == "":
            return "No ha seleccionado una imagen"

        if file and allowed_file2(filename=filename, allowed_set=allowed_set):
            #Leemos una imagen como un vector de numpy con RGB
            img = imread(name=file, mode='RGB')
            # Detecta y recorta una imagen de 160 x 160 que contiene un rostro
            img = get_face(img=img, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)

            # Si una cara es detectada
            if img is not None:

                embedding = forward_pass(
                    img=img, session=facenet_persistent_session,
                    images_placeholder=images_placeholder, embeddings=embeddings,
                    phase_train_placeholder=phase_train_placeholder,
                    image_size=image_size
                )

                embedding_dict = load_embeddings()
                if embedding_dict:
                    #Metodo para comparar la distancia euclideana entre este embedding y los embedding en la carpeta embeddings
                    identity = identify_face(embedding=embedding, embedding_dict=embedding_dict)
                    return render_template('resultadoprediccion.html', identity=identity)

                else:
                    return render_template(
                        'resultadoprediccion.html',
                        identity=" No se detectaron embedding. Por favor suba una imagen para detectar embeddings"
                    )

            else:
                return render_template(
                    'resultadoprediccion.html',
                    identity=" No se detectaron rostros"
                )
    else:
        return "Metodos POST HTTP es requerido"


@app.route("/envivo", methods=['GET', 'POST'])
def face_detect_live():
    #Detectar caras en el video de la webcam

    embedding_dict = load_embeddings()
    if embedding_dict:
        try:
            cap = cv2.VideoCapture(0)

            while True:
                cap.grab()  # For use in multi-camera environments when the cameras do not have hardware synchronization
                return_code, frame_orig = cap.read()  # Leemos el frame

                # Cambiamos de tamano el framo para mas rapido los calculos
                frame = cv2.resize(frame_orig, (0, 0), fx=0.5, fy=0.5)

                # Convertimos la imagen de BGR color ( OpenCV uses) to RGB color
                frame = frame[:, :, ::-1]

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if frame.size > 0:
                    faces, rects = get_faces_live(img=frame, pnet=pnet, rnet=rnet, onet=onet, image_size=image_size)

                    # Si una cara es detectada
                    if faces:
                        for i in range(len(faces)):
                            face_img = faces[i]
                            rect = rects[i]

                            # Scale coordinates of face locations by the resize ratio
                            rect = [coordinate * 2 for coordinate in rect]

                            face_embedding = forward_pass(
                                img=face_img, session=facenet_persistent_session,
                                images_placeholder=images_placeholder, embeddings=embeddings,
                                phase_train_placeholder=phase_train_placeholder,
                                image_size=image_size
                            )

                            #Metodo para comparar la distancia euclideana entre este embedding y los embedding en la carpeta embeddings
                            identity = identify_face(embedding=face_embedding, embedding_dict=embedding_dict)

                            cv2.rectangle(frame_orig, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

                            W = int(rect[2] - rect[0]) // 2
                            H = int(rect[3] - rect[1]) // 2

                            cv2.putText(frame_orig, identity, (rect[0]+W-(W//2), rect[1]-7),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                        cv2.imshow('Presiona la tecla "q" para cerrar ', frame_orig)
                
                    # Mostrar el video de la camara, sino se detectan rostros humanos
                    cv2.imshow('Presiona la tecla "q" para cerrar ', frame_orig)
                else:
                    continue

            cap.release()
            cv2.destroyAllWindows()
            return render_template('index.html')
        except Exception as e:
            print(e)
    else:
        return " Ningun embedding fue detectado, por favor sube una imagen para el embedding"


@app.route("/svmprediccion", methods=['GET', 'POST'])
def face_svm():
    img_path='./prueba_img/brandon.jpg'
    #uploads_path = os.path.join(APP_ROOT)
    modeldir = './modelo_transferlearning/20170511-185253.pb'
    classifier_filename = './resultados/classifier2.pkl'
    file = request.files['file']
    filename = 'prueba_img/' + file.filename
    file.save(filename)
    #img = imread(name=file, mode='RGB')
    #save_image(img=img, filename=filename, uploads_path=APP_ROOT)
    npy='./npy'
    train_img='./static/photos'
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 182
            input_image_size = 160
            
            HumanNames = os.listdir(train_img)
            HumanNames.sort()

            print('Cargando features')
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            c = 0



            print('Comienza reconocimiento')
            prevTime = 0
            # ret, frame = video_capture.read()
            frame = cv2.imread(filename,1)

            #frame = cv2.resize(frame, (0,0), fx=0.75, fy=0.75)    #resize frame (optional)
            #frame = cv2.resize(frame,None,fx=0.75,fy=0.75)


            curTime = time.time()+1    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Caras detectadas: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is too close')
                            continue

                        if(i>len(cropped)):
                            print('Running')
                            break
                        else:
                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])

                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                    
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            print(predictions)
                            best_class_indices = np.argmax(predictions, axis=1)
                            # print(best_class_indices)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            print(best_class_probabilities)
                            print(best_class_probabilities[0])
                            #if best_class_probabilities[0] < 0.5:
                            #    print("La persona no esta registrada")

                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                            text_2x = bb[i][0]
                            text_2y = bb[i][3] + 30

                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            print('Resultado en el vector: ', best_class_indices[0])
                            print(HumanNames)
                            if best_class_probabilities[0]:

                                for H_i in HumanNames:
                                    # print(H_i)
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        text = '{}: {:.2f}%'.format(result_names, best_class_probabilities[0]*100)
                                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 255), thickness=1, lineType=2)
                            #else:
                                #cv2.putText(frame, 'Desconocido' , (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    print('Unable to align')
            cv2.imshow('Presione la tecla "q" para salir ', frame)
            #return render_template('index.html')

            if cv2.waitKey(100000) & 0xFF == ord('q'):
                #sys.exit("Gracias")
                cv2.destroyAllWindows()
            return render_template('index.html')


@app.route("/videosvm", methods=['GET', 'POST'])
def video_svm():
    #input_video="akshay_mov.mp4"
    modeldir = './modelo_transferlearning/20170511-185253.pb'
    classifier_filename = './resultados/classifier2.pkl'
    npy='./npy'
    train_img='./static/photos'

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 182
            input_image_size = 160
            
            HumanNames = os.listdir(train_img)
            HumanNames.sort()

            print('Loading Modal')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            video_capture = cv2.VideoCapture(0)
            c = 0


            print('Comieza el reconocimiento')
            prevTime = 0
            while True:
                ret, frame = video_capture.read()

                #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

                curTime = time.time()+1    # calc fps
                timeF = frame_interval

                if (c % timeF == 0):
                    find_results = []

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('Nro de Caras detectadas: %d' % nrof_faces)

                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces,4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('Face is very close!')
                                continue
                            if(i>len(cropped)):
                                print('Running')
                                break
                            else:
                                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                                cropped[i] = facenet.flip(cropped[i], False)
                                scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                                scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled[i] = facenet.prewhiten(scaled[i])
                                scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                print(predictions)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                # print("predictions")
                                print(best_class_indices,' with accuracy ',best_class_probabilities)

                                # print(best_class_probabilities)
                                #if best_class_probabilities[0]:
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                #plot result idx under box
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                print('Resultado Indice Vector: ', best_class_indices[0])
                                print(HumanNames)
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        #text = "{:.2f}%".format(best_class_probabilities*100)
                                        text = '{}: {:.2f}%'.format(result_names, best_class_probabilities[0]*100)
                                        cv2.putText(frame,text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 255), thickness=1, lineType=2)
                                #else:
                                #    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                                    #plot result idx under box
                                #    text_x = bb[i][0]
                                #    text_y = bb[i][3] + 20
                                #    print('Resultado Indice Vector: ', best_class_indices[0])
                                #    print(HumanNames)
                                #    for H_i in HumanNames:
                                #        if HumanNames[best_class_indices[0]] == H_i:
                                #            result_names = HumanNames[best_class_indices[0]]
                                #            cv2.putText(frame, 'Desconocido', (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #                        1, (0, 0, 255), thickness=1, lineType=2)
                                
                                
                    else:
                        print('Alignment Failure')
                # c+=1
                cv2.imshow('Presione la tecla "q" para cerrar ', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            cv2.destroyAllWindows()
            return render_template('index.html')

@app.route("/recortar", methods=['GET','POST'])
def recortar():
    input_datadir = './static/photos'
    output_datadir = './static/pre_img'
    obj=preprocesses(input_datadir,output_datadir)
    nrof_images_total,nrof_successfully_aligned=obj.collect_data()

    print('El numero total de imagenes es: %d' % nrof_images_total)
    print('Numero total de imagenes alineadas: %d' % nrof_successfully_aligned)

    return render_template('recortado.html', nrof_images_total=nrof_images_total)

@app.route("/entrenamiento2", methods=['GET','POST'])
def entrenarte():
    #Entrenamiento
    datadir = './static/pre_img'
    modeldir = './modelo_transferlearning/20170511-185253.pb'
    classifier_filename = './resultados/classifier2.pkl'
    print ("Entrenando")
    obj=training(datadir,modeldir,classifier_filename)
    get_file=obj.main_train()
    print('Guardado en la carpeta resultados "%s"' % get_file)

    # sys.exit("Termino el entrenamiento")
    return render_template('entrenado.html')


@app.route("/entrenamientotransfer", methods=['GET','POST'])
def entrenartransfer():
    # load images in grayscale
    imgs, usr = load_imgs('Faces/')
    # create labels
    yy = create_labels(imgs)
    # reshape data in keras format
    xx = reshape_for_keras(imgs)
    # normalize data
    xx = xx/255 
    # divide train/test
    train_index, test_index = train_test_split(xx, test_size=0.2)
    x_train = xx[train_index]
    y_train = yy[train_index]
    x_test = xx[test_index]
    y_test = yy[test_index]
    # print useful info
    print('Total number of samples:', xx.shape[0])
    print('\tNumber of training samples:', x_train.shape[0])
    print('\tNumber of test samples:', x_test.shape[0])
    # create keras model
    cl = create_model(xx, len(usr))
    train_and_evaluate(cl, x_train, y_train, x_test, y_test, len(usr))
    # save trained model
    save_model(cl)
    # display some features extracted from the CNN
    return render_template('entrenado.html')


@app.route("/transfer", methods=['GET','POST'])
def webimagenet():
    # face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cl = load_model2()
    cap = cv2.VideoCapture(0)
    predicted_proba = 0.0
    predicted_name = ''
    while True:
        grabbed, frame = cap.read()	
        if grabbed:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            face_imgs, faces = detect_face2(frame, face_cascade)
            if len(faces) > 0:
                predicted_labels, predicted_probas = face_classification(face_imgs, cl)
                predicted_name = map_label(predicted_labels)
                print('------------')
                for i,j in zip(map_label(predicted_labels), predicted_probas):	
                    print('Detectado %s w.p. %f' % (i,j))
            display_result(frame, faces, predicted_name)
    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')

def process_frame(frame, face_classifier):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.6, 8)

	return faces


def display_result2(img, faces, save_results, n_img):
    if request.method == "POST":
        num = 0
        max_num=1000
        for (x,y,w,h) in faces:
            num += 1
            if num == max_num:
                break

            # if detected save face in .png
            if save_results and len(faces)==1:
                t_index = datetime.datetime.now()
                t_index = "%s_%s_%s_%s" % (t_index.hour, t_index.minute, t_index.second, str(t_index.microsecond)[:2])
                mask = img[y:y+h,x:x+w,:]
                cv2.imwrite(FACES_FOLDER + '/' + request.form['fullname2'] + '/' + request.form['fullname2'] + '_' + t_index + '.png', mask)

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(img, ('%d' % n_img), (x+50,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 4)
        # display results
        cv2.imshow("Presiona la tecla 'q' para cerrar ", img)
        cv2.waitKey(1)

@app.route('/agregar', methods=['POST'])
def agregar():
    if request.method == "POST":
        max_num=1000
        if SAVE_FACE and not os.path.exists(FACES_FOLDER + '/' +   request.form['fullname2']):
	        os.makedirs(FACES_FOLDER + '/' + request.form['fullname2'])
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        n_img = 0
        while True:
            grabbed, frame = cap.read()
            if grabbed:
                if n_img == max_num:
                    break
                faces = process_frame(frame, face_cascade)
                if len(faces) != 0:
                    if SAVE_FACE:
                        n_img+=1
                        print('# guardando img:', n_img)
                    else:
                        print('Cara detectada')
                display_result2(frame, faces, SAVE_FACE, n_img)

        cap.release()
        cv2.destroyAllWindows()
        return render_template('index.html')
    else:
        return "Metodo Post Requerido"



@app.route("/tomarfoto", methods=['GET','POST'])
def tomarfoto():
    subprocess.run('start microsoft.windows.camera:', shell=True)
    return render_template('index.html')


@app.route("/menui")
def index_page():
    """Renderiza la 'index.html' pagina para subir las imagenes para crear el embedding."""
    return render_template("index.html")


@app.route("/prediccion")
def predict_page():
    """Renderiza la 'prediccion.html' pagina para subir las imagenes para predecir."""
    return render_template("prediccion.html")

@app.route("/menu")
def menu_page():
    """Renderiza la 'menu.html' pagina al menu."""
    return render_template("menu.html")

@app.route("/agregarpersona")
def add_person():
    """Renderiza la 'nuevapersona.html' pagina al nueva persona."""
    return render_template("nuevapersona.html")

@app.route("/indexsvmmmmmm")
def index_svm():
    """Renderiza la 'nuevapersona.html' pagina al nueva persona."""
    return render_template("indexsvm.html")




if __name__ == '__main__':
    """Server and FaceNet Tensorflow configuration."""

    #Cargando el modelo de Facenet y configurando los placeholders para el forward pass en el Facenet modell para calcular los embeddings
    model_path = 'model/20170512-110547/20170512-110547.pb' #Este es Inception Model
    #modeldir = './modelo_transferlearning/20170511-185253.pb' #El otro modelo inception
    #facenet_model2 = load_model(modeldir)

    facenet_model = load_model(model_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    image_size = 160 #tamano de la imagen de input 160x160x3
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Initiate persistent FaceNet model in memory
    #Iniciando el modelo de Facenet en memoria
    facenet_persistent_session = tf.Session(graph=facenet_model, config=config)

    #Creando Multi-Task Cascading Convolutional (MTCNN) para la deteccion de caras
    pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)

    # Iniciando el Servidor de Flask
    #serve(app=app, host='0.0.0.0', port=5000)
    app.run(debug=True)
