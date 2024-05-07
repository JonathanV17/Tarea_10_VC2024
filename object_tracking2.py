'''
Nombre del archivo: deteccion_objetos.py 
Descripcion:    Script en el que se realiza la detección de objetos, usando
                segmentación, y caracterización de objetos, con la finalidad
                de llevar un control del objeto de las veces que pasa o cruza
                cierta region, así como su tiempo en cada región

Autor(es):  Jonathan Ariel Valadez Saldaña

Fecha de creación: 6/Mayo/2024

Ejemplo de uso:   python object_tracking.py --img_object Imagen_cinta.jpeg --video Video_Cinta.mov

'''
import cv2              
import argparse         
import numpy as np      
import sys 
import time
from typing import Tuple

# Límites inferiores y superiores de los valores del HSV
#lower_range = np.array([85, 60, 0])  # Lower HSV threshold
#upper_range = np.array([180, 180, 70])  # Upper HSV threshold
lower_range = np.array([95, 100, 0])  # Lower HSV threshold
upper_range = np.array([180, 180, 80])  # Upper HSV threshold

def parser_user_data()-> argparse:
    """
    Descripción:    Función que recibe la información de entrada por el usuario
    
    Parámetros:     Ninguno

    Regresa:        args(argparse) objeto con la información o datos recibidos 
    """
    parser = argparse.ArgumentParser(description='Control de detección de un objeto en video.')
    parser.add_argument('--img_object', required=True, 
                        help='Introducir el nombre del archivo de la imagen del objeto de interés')
    parser.add_argument('--video', required=True,
                        help='Introducir el nombre del archivo del video con el objeto de interés')
    args = parser.parse_args()
    return args

def load_files(filename_img:argparse, filename_video:argparse) -> Tuple[str, cv2.VideoCapture]:
    """
    Descripción:    carga dos archivos (imagen y video) desde la ruta especificada
    
    Parámetros:     filename_img(str): ruta del archivo de la imagen
                    filename_video(str): ruta del archivo del video

    Regresa:        imagen(str) y video(cv2.VideoCapture) siendo los archivos cargados
    """
    imagen = filename_img
    video = cv2.VideoCapture(filename_video)

    if video is None:
        print("Error: The video file failed to load. Check the file path.")
        exit()

    return imagen, video


def segment_and_characterization(cap: cv2.VideoCapture, imagen: str) -> None:
    """
    Descripción:    Funcion que segmenta la imagen y video para aplicarles caracterización 
                    de puntos y descriptores, así como detectar el objeto de interés 

    Parámetros: cap(cv2) e imagen(str)

    Regresa: nada
    """
    # Crear objeto orb para 25 características
    orb = cv2.ORB_create(nfeatures=25)

    # Cargar la imagen de referencia 
    img = cv2.imread(imagen)

    # Inicializar variables de tiempo
    start_time = None
    left_time = 0
    right_time = 0
    left_to_right = 0
    right_to_left = 0
    
    while cap.isOpened():
        # Leer el frame actual
        ret, frame = cap.read()

        # Si no se pudo leer un frame, desplegar mensaje de error
        if not ret:
            print("ERROR! - el frame último no pudo ser leído")
            break
        
        # FILTRO PARA VIDEO ------------------------------------------------------------------

        # Aplicamos filtro de la mediana a los frames, con tamaño de kernel de 35
        frame_median = cv2.medianBlur(frame, 35)

        # Convertimos los frames filtrados, de BFR a HSV
        frame_HSV = cv2.cvtColor(frame_median, cv2.COLOR_BGR2HSV)

        # Aplicamos los límites de los valores del HVS a los frames
        frame_threshold = cv2.inRange(frame_HSV, lower_range, upper_range)

        # Resaltamos la región de interés (1-blanco) e ignoramos lo demás (0-negro)
        bitwise_AND = cv2.bitwise_and(frame, frame, mask=frame_threshold)

        # Aplicamos escala de grises a bitwise_AND
        frame_gray = cv2.cvtColor(bitwise_AND, cv2.COLOR_BGR2GRAY)

        # FILTRO PARA IMAGEN -----------------------------------------------------------------

        # Aplicamos filtro de la mediana a la imagen, con tamaño de kernel de 35
        img_median = cv2.medianBlur(img, 35)

        # Convertimos la imagen filtrada, de BFR a HSV
        img_HSV = cv2.cvtColor(img_median, cv2.COLOR_BGR2HSV)

         # Aplicamos los límites de los valores del HVS a la imagen
        img_threshold = cv2.inRange(img_HSV, lower_range, upper_range)

        # Resaltamos la región de interés (1-blanco) e ignoramos lo demás (0-negro)
        bitwise_AND2 = cv2.bitwise_and(img, img, mask=img_threshold)

         # Aplicamos escala de grises a bitwise_AND2
        img_gray = cv2.cvtColor(bitwise_AND2, cv2.COLOR_BGR2GRAY)

        # ------------------------------------------------------------------------------------

        # Detectar puntos clave en la imagen de referencia utilizando goodFeaturesToTrack
        kp_ref = cv2.goodFeaturesToTrack(img_gray, maxCorners=50, qualityLevel=0.01, minDistance=10)

        # Crear una lista de objetos cv2.KeyPoint para la imagen de referencia
        kp_ref1 = []
        for point in kp_ref:
            x_coordinate = point[0][0]
            y_coordinate = point[0][1]

            # Crear un objeto cv2.KeyPoint con las coordenadas
            keypoint = cv2.KeyPoint(x_coordinate, y_coordinate, 2)

            # Agregar el objeto cv2.KeyPoint a la lista kp_ref1
            kp_ref1.append(keypoint)

        # Detectar puntos clave en el frame del video utilizando goodFeaturesToTrack
        kp_frame = cv2.goodFeaturesToTrack(frame_gray, maxCorners=50, qualityLevel=0.01, minDistance=10)

        # Crear una lista de objetos cv2.KeyPoint para el frame del video
        kp_frame1 = []
        for point in kp_frame:
            x_coordinate = point[0][0]
            y_coordinate = point[0][1]

            # Crear un objeto cv2.KeyPoint con las coordenadas
            keypoint = cv2.KeyPoint(x_coordinate, y_coordinate, 2)

            # Agregar el objeto cv2.KeyPoint a la lista kp_frame1
            kp_frame1.append(keypoint)

        # Calcular descriptores para los puntos clave de la imagen de referencia utilizando ORB
        kp1, des1 = orb.compute(img_gray, kp_ref1)

        # Calcular descriptores para los puntos clave del frame del video utilizando ORB
        kp2, des2 = orb.compute(frame_gray, kp_frame1)

        # Creamos objeto bf 
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Unimos los descriptores de la imagen y el video
        matches = bf.match(des1, des2)

        # Dibujamos todas las coincidencias
        frame_matches = cv2.drawMatches(img, kp1, frame_gray, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # ------------------------------------------------------------------------------------
    
        # Buscamos los contornos en los límites del frame
        contours, _ = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujamos el rectángulo alrededor del objeto detectado
        for contour in contours:
            # Obtenemos las cordenadas del vertice superior izquierdo (x,y), y sus dimensiones (w,h)
            x, y, w, h = cv2.boundingRect(contour)

            # Calcular el momento de orden cero para el área del rectángulo
            M = cv2.moments(contour)

            # Evitar divisiones por cero
            if M["m00"] != 0:

                # Calcular el centroide del rectángulo
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])

                # Dibujar el centroide
                cv2.circle(frame, (centroid_x, centroid_y), 5, [0, 0, 255], -1)

                # Dibujar el rectángulo alrededor del objeto
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Dibujar puntos de coincidencia en la ventana 'Detected Objects'
            for match in matches:
                #(x1, y1) = kp1[match.queryIdx].pt
                (x2, y2) = kp2[match.trainIdx].pt
                cv2.circle(frame, (int(x2), int(y2)), 5, (255, 255, 0), -1)

        # Dibujar una línea verde diagonal de izquierda a derecha del video
        #cv2.line(frame, (0, frame.shape[0] ), (frame.shape[1], 0), (0, 255, 0), 2)
        cv2.line(frame, (0 + 500, frame.shape[0] ), (frame.shape[1] - 500, 0), (0, 255, 0), 2)

        # Llamar a la función para contar cruces y actualizar tiempos
        start_time, left_time, right_time, left_to_right, right_to_left = crossCounting_and_time(centroid_x, centroid_y, frame, start_time, left_time, right_time, left_to_right, right_to_left)

        # Dibujar texto con información sobre tiempos y cruces en el fotograma
        frame = draw_text_on_frame(frame, left_time, right_time, left_to_right, right_to_left)

        # Definir el tamaño de las ventanas (ancho, alto)
        window_w = 700 
        window_h = 480

        # Aplicar el tamaño de las ventanas
        cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Matches', window_w, window_h)
        cv2.namedWindow('Detection video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detection video', window_w, window_h)

        # Mostrar el marco con las coincidencias
        cv2.imshow('Matches', frame_matches)

        # Mostrar el fotograma con los objetos detectados
        cv2.imshow('Detection video', frame)
        
        # Cerrar el programa al precionar 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Programa finalizado!")
            break

def crossCounting_and_time(centroid_x, centroid_y,frame, start_time, left_time, right_time, left_to_right, right_to_left):
    """
    Descripción:    Funcion que realiza los conteos de cruces de izq-der y der-izq, así como
                    el tiempo en cada lado

    Parámetros: frame(cv2), left_time(int), right_time(int), left_to_right(int), 
                right_to_left(int), start_time(float), centroid_x(int), variables
                inicializadas en 0, excepto centroid_x, el cual ya tiene un valor
                calculado previamente

    Regresa: Todos los parámetros ya contabilizados, excepto centroid_x
    """
    # Coordenadas de los puntos inicial y final de la línea
    x1, y1 = 0 + 500, frame.shape[0]
    x2, y2 = frame.shape[1] - 500, 0

    # Calculamos la pendiente
    m = (y2 - y1) / (x2 - x1)

    # Calculamos el término de intersección
    b = y1 - m * x1

    # Calculamos el valor de Y de nuestra línea diagonal
    y = m * centroid_x + b

    # Centroide Y debajo de la diagonal 
    if centroid_y > y:
        if start_time is None:
            start_time = time.time()
        right_time += (time.time() - start_time)/1000

        if left_time > 0 and left_to_right == 0 and right_to_left == 0:
            left_to_right += 1
        elif left_time > 0 and right_to_left > left_to_right:
            left_to_right += 1
        #elif left_time > 0 and right_to_left == left_to_right:
        #    left_to_right += 1

    # Centroide Y arriba de la diagonal
    else:
        if start_time is None:
            start_time = time.time()
        left_time += (time.time() - start_time)/1000

        if right_time > 0 and right_to_left == 0 and left_to_right == 0:
            right_to_left += 1  
        elif right_time > 0 and right_to_left == left_to_right:
            right_to_left += 1 
        #elif right_time > 0 and left_to_right > right_to_left:
        #    right_to_left += 1 
         
    return start_time, left_time, right_time, left_to_right, right_to_left

def draw_text_on_frame(frame, left_time, right_time, left_to_right, right_to_left):
    """
    Descripción:    Funcion que imprime o dibuja los valores de los parámetros calculados 
                    previamente sobre cada frame del video

    Parámetros: frame(cv2), left_time(int), right_time(int), left_to_right(int), 
                right_to_left(int), start_time(float), se van a tomar los valores que 
                toman estos parámetros en cada frame

    Regresa: frame(cv2), ya con los valores de los parámetros dibujados en el video
    """

    # Dividir el texto en varias líneas
    text_line1 = f"Tiempo izq: {left_time:.2f} s"
    text_line2 = f"Tiempo der: {right_time:.2f} s"
    text_line3 = f"Cruces izq-der: {left_to_right}"
    text_line4 = f"Cruces der-izq: {right_to_left}"

    # Definir la posición vertical de cada línea de texto
    text_position_line1 = (20, 40)
    text_position_line2 = (20, 80)
    text_position_line3 = (20, 120)
    text_position_line4 = (20, 160)

    # Configurar el texto y la posición en el fotograma para cada línea
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (0, 0, 0)  # Color negro

    # Dibujar cada línea de texto en el fotograma
    cv2.putText(frame, text_line1, text_position_line1, font, font_scale, text_color, font_thickness)
    cv2.putText(frame, text_line2, text_position_line2, font, font_scale, text_color, font_thickness)
    cv2.putText(frame, text_line3, text_position_line3, font, font_scale, text_color, font_thickness)
    cv2.putText(frame, text_line4, text_position_line4, font, font_scale, text_color, font_thickness)

    return frame

def close_windows(cap:cv2.VideoCapture)->None:
    """
    Descripción: Funcion que cierra las ventanas una vez se haya frenado el procesamiento de las mismas

    Parámetros: cap(cv2)

    Regresa: Nada
    """
    # Cerramos todas las ventanas
    cv2.destroyAllWindows()

    # Finalizamos cámara (Importante hacerlo)
    cap.release()                       

def run_pipeline():
    """
    Descripción: Funcion que se encarga se ejecutar las demás funciones

    Parámetros: Ninguno

    Regresa: Nada
    """
    # Captura del argumento
    args = parser_user_data()

    # Inicialización de la cámara
    imagen, video = load_files(args.img_object, args.video)

    # Segmentación de objetos
    segment_and_characterization(video, imagen)

    # Cerrar ventanas
    close_windows(video)

if __name__ == "__main__":
    run_pipeline()
