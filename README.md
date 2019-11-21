# Tesis Reconocimiento facial Facenet, SVM y Transfer Learning

Aplicacion web desarrollada para mostrar los requerimientos de trabajo de grado para Ingeniero de sistemas titulado "Sistema de Reconocimiento Facial usando Redes Neuronales Convolucionales". 

Se puede evidenciar un resumen de los lenguajes de programación, herramientas y librerías de todo el sistema, incluyendo Modelos de Deep Learning, API REST y Web App. Primero existe la capa de los modelos de Deep Learning los cuales son desarrollados en Python utilizando las principales librerías de ciencias de Datos e Inteligencia Artificial: Tensorflow, Keras, Scikit-learn y Numpy. Además, se le añade la librería de visión Computarizada OpenCV. Seguidamente sigue el servidor backend que funciona como API REST desarrollado en el framework de Python Flask. Por último, se encuentra la capa de Frontend e interfaz gráfica, desarrollada utilizando herramientas de diseño web como lo son HTML, CSS y el lenguaje de programación Javascript, y librerías como Bootstrap y JQuery.  Finalmente, el usuario final podrá acceder a la aplicación desde cualquier navegador web. Ver La Siguiente Imagen

![Repo List](Screens/arquitectura.png)

## Fases
Se desarrolló el algoritmo siguiendo las fases descritas por Li y Jain (2011).

 * Detección de caras: 
   Para detección de caras actualmente los dos algoritmos más usados son MTCNN ( Multi-task Cascaded Convolutional Networks) y Haar Cascades de la librería OpenCV. El archivo  Haar Cascades se encuentra en este repo llamado haarcascade_frontalface_default.xml. Para descargar los pesos del MTCNN crear una carpeta llamada "npy" y colocar los 3 archivos "det1.npy", "det2.npy", "det3.npy"
   
 * Acondicionamiento y normalización de caras: 
   En segundo lugar, luego que la cara es detectada, se recortan todas las imágenes del mismo tamaño y alinea para ser guardada en la base de datos. Dicha acción se realizó con funciones de la librería OpenCV para el preprocesamiento de imágenes
   
 *  Comparación de características: 
  En tercer lugar, es necesario resaltar las características únicas de la cara, que se pueden usar para distinguirla de otras personas. ( Li y Jain, 2011) Para realizar la extracción de característica se utilizó la técnica de deep learning, específicamente redes neuronales convolucionales preentrenadas .Unos de los modelos preentrenados utilizados en dos de las soluciones implementadas es
FaceNet utlizando la librería Tensorflow También se realizó la técnica de transfer learning entrenado la capa de clasificación softmax utilizando la Librería de Keras, los pesos de los datasets de Imagenet y VGGFace.

## Modelo con ANN de clasificador
  
![Repo List](Screens/keras.png)

El primer modelo se basa en añadir una segunda red neuronal de clasificación con una capa de clasificación categórica softmax, luego de la red neuronal convolucional que extrae las características de la cara. Dada una imagen de una persona, dicha capa softmax retorna la probabilidad de cada persona guardada en la base de datos con respecto a la similitud de la imagen. En la anterior figura se puede ver una arquitectura de este modelo y las tecnologías de desarrollo de cada uno. 

## Modelo utilizando One Shot Learning
  
![Repo List](Screens/one.png)

El segundo modelo consiste en dado el vector de características de 128 dimensiones extraído del modelo de red neuronal convolucional de FaceNet, comparar las distancias euclidianas de este vector con los vectores guardados en la base de datos. Este modelo se conoce con el nombre de one shot learning, debido a que solo requiere una imagen por persona. 


## Modelo  utilizando SVM de clasificador
  
![Repo List](Screens/svm.png)

El tercer modelo que se realizó consiste en colocar un algoritmo de aprendizaje supervisado llamado SVM (Support-Vector Machine) al final del modelo de red neuronal convolucional de FaceNet. Se utilizó la librería de Python llamada
Scikit-learn para implementar el algoritmo. 

## Requerimientos
Tener instalados
* Python 3.6
* Tensorflow 
* Keras
* OpenCV
* Numpy
* Scikit-learn

* Las versiones y otras librerias estan en *requirements.txt*.

*Para instalarlas usar el comando  ```pip install -r requirements.txt```

* Se recomienda crear un enviroment de python donde se instalan las liberias con sus correspondientes versiones.

## Steps
1. Descargar el modelo preentrenado de aqui (https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit).

2. Mover el archivo descargado 'model/' folder, de la siguiente manera:

     ```'model/20170512-110547/20170512-110547.pb'```
     
3. Descargar el otro modelo preentrenado de :

4. Crear una carpeta llamada "npy" y colocar los 3 archivos "det1.npy", "det2.npy", "det3.npy" descargados:

5. Crear una carpeta llamada "Faces", aqui se guardan las fotos que captura el modelo de transfer learning.

6. Correr el siguiente comando en la consola ```python server.py```.

7. Abrir la URL en el navegador ( 127.0.0.1:5000).


## Repositorios de los papers de Referencia
* Facenet: [paper](https://arxiv.org/abs/1503.03832) - [repository](https://github.com/davidsandberg/facenet)

* Multi-Task Cascading Convolutional Neural Network (MTCNN) for face detection: [paper](https://arxiv.org/abs/1604.02878) - [repository](https://github.com/foreverYoungGitHub/MTCNN)

