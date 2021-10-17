#Codigo realizado por Sebastian Marinho y Daniel Barandica, para la materia de Procesamiento de imagenes y video

# Se definen las librerias necesarias
import numpy as np
import cv2
import os


## Se definen algunas variables
points = [] ## Los puntos seleccionados por el usuario en las imagenes
H_list = [] ## La lista donde se guardaran las H
concat = [] ## Imagenes donde se guardara la concatenacion
flag = False


#Funcion recibir para pedir el path de las imagenes
def recibir():
    path = input("Ingrese la dirección de la carpeta donde se encuentras sus imagenes: ") #Se pide el path
    imagenes = [] #Lista para guardar imagenes
    cont = 1 #Contador
    metodo = input("Presione 1 si desea usar SIFT y 2 para usar ORB: ")
    while (True):
        try:
            #Se guardan las imagenes que estan en el path ingresado y que tengan como nombre image_# y formato JPEG
            image_name = "image_" + str(cont) + ".jpeg"
            path_file = os.path.join(path, image_name)
            image = cv2.imread(path_file)
            image = cv2.resize(image, (900, 980))
            imagenes.append(image)
            cont += 1
        except:
            break
    N = len(imagenes) #Numero de imagenes leidas
    return imagenes, N, metodo


#Funcion click para guardar los puntos que son puestos por cada usuario
def click(event, x, y, flags, param):
    global flag
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        flag = True

#Funcion Homography que fue tomada y adaptada de codigo realizado por Julian Quiroga
def Homography(image, image_2,metodo):
    image_1 = image
    image_1 = cv2.resize(image_1, (900, 980))
    image_gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_draw_1 = np.copy(image_1)
    image_2 = image_2
    image_2 = cv2.resize(image_2, (900, 980))
    image_gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    image_draw_2 = np.copy(image_2)

    # sift/orb interest points and descriptors
    if int(metodo) == 1:
        sift = cv2.SIFT_create(nfeatures=10000)  # shift invariant feature transform
        keypoints_1, descriptors_1 = sift.detectAndCompute(image_gray_1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(image_gray_2, None)
        image_draw_1 = cv2.drawKeypoints(image_gray_1, keypoints_1, None)
        image_draw_2 = cv2.drawKeypoints(image_gray_2, keypoints_2, None)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(descriptors_1, descriptors_2, k=1)
        image_matching = cv2.drawMatchesKnn(image_1, keypoints_1, image_2, keypoints_2, matches, None)
        met = "Sift"
        print("Sift")
    else:
        orb = cv2.ORB_create(nfeatures=10000)  # oriented FAST and Rotated BRIEF
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_gray_1, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_gray_2, None)
        image_draw_1 = cv2.drawKeypoints(image_gray_1, keypoints_1, None)
        image_draw_2 = cv2.drawKeypoints(image_gray_2, keypoints_2, None)
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = bf.match(descriptors_1,descriptors_2)
        # image_matching = cv2.drawMatches(image_1, keypoints_1, image_2, keypoints_2, matches[:10], flags=2)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(descriptors_1, descriptors_2, k=1)
        image_matching = cv2.drawMatchesKnn(image_1, keypoints_1, image_2, keypoints_2, matches, None)
        met= "orb"
        print("orb")

    # Retrieve matched points
    points_1 = []
    points_2 = []
    des_1 = []
    des_2 = []
    for idx, match in enumerate(matches):
        idx2 = match[0].trainIdx

        dif = np.sum(np.power(descriptors_1[idx] - descriptors_2[idx2], 2))
        if (dif < 10000):
            points_1.append(np.int32(keypoints_1[idx].pt))
            points_2.append(np.int32(keypoints_2[idx2].pt))
            des_1.append(descriptors_1[idx])
            des_2.append(descriptors_2[idx2])

    # Compute homography and warp image_1
    H, _ = cv2.findHomography(np.array(points_1), np.array(points_2), method=cv2.RANSAC)
    return H,met


#Funcion promedio_images para promediar imagenes
def promedio_imagenes(img_1, img_2):
    #Binarizacion de las dos imagenes de entrada
    _, Ibw_1 = cv2.threshold(img_1[..., 0], 1, 255, cv2.THRESH_BINARY)
    _, Ibw_2 = cv2.threshold(img_2[..., 0], 1, 255, cv2.THRESH_BINARY)

    #Operacion And entre las imagenes binarizada
    mask = cv2.bitwise_and(Ibw_1, Ibw_2)

    #Operacion And entre mask y la imagen de entrada original a color
    img_1_l = cv2.bitwise_and(img_1, cv2.merge((mask, mask, mask)))
    img_2_l = cv2.bitwise_and(img_2, cv2.merge((mask, mask, mask)))


    #Conversion a uint32
    img_2_l = np.uint32(img_2_l)
    img_1_l = np.uint32(img_1_l)

    #Suma de ambas imagenes y division entera sobre 2
    img = np.uint8((img_2_l + img_1_l) // 2)

    #Mascara negada
    n_mask = cv2.bitwise_not(mask)

    #Operacion And entre n_mask y la imagen de entrada original a color
    img_1 = cv2.bitwise_and(img_1, cv2.merge((n_mask, n_mask, n_mask)))
    img_2 = cv2.bitwise_and(img_2, cv2.merge((n_mask, n_mask, n_mask)))

    #Operacion or entre las nuevas mascaras e img
    img = cv2.bitwise_or(img, img_1)
    img = cv2.bitwise_or(img, img_2)
    return img #Imagen promediada


def Recortar(imagen): ## Se crea la funcion recortar para eliminar el exceso de bits negros en la imagen
    puntos = np.where((imagen[:,:,0]>0) * (imagen[:,:,1]>0)* (imagen[:,:,2])>0)
    if(puntos[0].shape[0]>0):
                max_y = max(puntos[0])
                max_x = max(puntos[1])
                min_y = min(puntos[0])
                min_x = min(puntos[1])
                img_guar = imagen[min_y:max_y,min_x:max_x,:]
                return img_guar
    else:
        return -1



#Funcion main
if __name__ == '__main__':
    #Se le pide al usuario ingresar el path de las imagenes y el numero de la imagen de referencia
    imagenes, N, metodo = recibir()
    print("El numero de imagenes recibidas es" + " " + str(N))
    ref = input("Escoja el numero de imagen de referencia: ")
    assert int(ref) <= N

    #Se realiza la homografia a cada una de las imagenes
    for i in range(N-1):
        a,met = Homography(imagenes[i], imagenes[(i + 1) % len(imagenes)],metodo)
        H_list.append(a)


    referencia = int(ref)-1 # Indice de imagen de referencia
    factor = 10 #Factor de escalado de la imagen
    des = 2200 #Factor de desplazamiento de la imagen

    h_traslacion = np.array([[1, 0, des], [0, 1, des], [0, 0, 1]], np.float64) #Matriz de desplazamiento

    img_transform = [] #Se guardan las imagenes de las diferentes perspectivas respecto a la referencia

    #Union de las homografias para creacion de imagen panoramica
    for i in range(N):
        h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64) #Matriz identidad
        if i > referencia: #Se evalua para tomar las imagenes de la derecha de la referencia
            for cont, j in enumerate(H_list[referencia:i]):
                h = j @ h
            h = np.linalg.inv(h) #Al estar en la derecha se debe realizar la inversa
        elif i < referencia: #Se evalua para tomar las imagenes de la izquierda de la referencia
            for j in (H_list[i:referencia]):
                h = h @ j


        if i != referencia: #Se evalua que la imagen no sea la de referencia
            #Se proyecta las imagenes de entrada a la perspectiva de la referencia transalada
            img_warp = cv2.warpPerspective(imagenes[i], h_traslacion @ h,
                                           (imagenes[0].shape[1] * (factor), imagenes[0].shape[0] * (factor)))

        else:
            #Se traslada la imagen de referencia
            img_warp = cv2.warpPerspective(imagenes[i], h_traslacion,
                                           (imagenes[0].shape[1] * (factor), imagenes[0].shape[0] * (factor)))

        img_transform.append(img_warp)#Se guardan las imagenes obtenidas de warp

    prom = np.zeros_like(img_transform[i]) #Se crea una matriz de ceros
    for idx, img in enumerate(img_transform):
        prom = promedio_imagenes(prom, img)  #Promedio entre las imagenes obtenidas de la homografia

    prom=Recortar(prom) ## Se recorta la imagen para una mejor visualizacion
    cv2.imwrite("Imagen_panoramica" + met +".png", prom) #Se muestra la imagen resultante en pantalla
    cv2.waitKey(0)