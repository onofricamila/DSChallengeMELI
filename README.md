# DSChallengeMELI

## 📌 Requerimientos
Una vez clonado el repositorio, es necesario:

- instalar [python](https://www.python.org/downloads/)
- abrir una terminal, ubicarse dentro de la carpeta **DSChallengeMELI** y correr `pip install -r requirements.txt`
- correr el script `project_set_up.py`: genera la estructura de carpetas necesaria para que no se generen inconvenientes al insertar dentro de los notebooks en carpetas inexistentes 

## 📌 Notas

- Los notebooks asociados a cada inciso estan numerados indicando el orden en que debería correrse cada uno
- Ciertas decisiones se tomaron teniendo en cuenta:
  - poder de cómputo (16Gb RAM, procesador i5)
  - no quería demorarme demasiado en la entrega de la resolución 
 
Es por eso que se listan en éste documento algunas de las ideas que se me ocurrieron y que podría probar en cada caso.
 

## 📌 Exploracion y analisis
En esté inciso, al traer los datos, aproveche para dejar las columnas que me interesaban para el siguiente 

En lo que respecta al armado del data set, podría haber trabajado con ...
  - información de cada producto del catálogo en sí (por ejemplo ratings, como usé los de vendedores)
  - las coordenadas de las ciudades, para agregar features de longitud/latitud, y poder representar mejor la ubicación geográfica (Ej: https://api.mercadolibre.com//classified_locations/cities/TUxBQkZMTzg5MjFa)

Además, podría haber aplicado algún método de [selección de features](https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/). Lo que hice fue dejar aquellas que me parecía serían de utilidad, considerando que siempre le dedico más tiempo a ver los datos y adquirir conocimiento del dominio. Por ejemplo, podría haber realizado una matriz de correlación con las features numéricas, para ver si alguna estaba fuertemente correlacionada con el target.


## 📌 Modelo

### Data set

Cuando hice la selección de features en el inciso anterior, dejé algunas relacionadas a tags y la que tiene el listado con todas las categorías desde la root, para cada item. Después, en el notebook donde separo en train y test, las quité.

En un principio las había seleccionado ya que quería provar codificar sus valores con embeddings. Para el caso de los tags, primero tendría que ver si hay muchos tags diferentes y distintas combinaciones como para que amerite a usar este enfoque. Luego, para lo que es el camino de todas las categorías, me parecía más intuitivo: me interesaba poder usar como feature no solo la categoria root, sino todas; para ésto, no resulta viable el uso de dummies: hay demasiadas categorías, y además, me gustaba el hecho de modelar la relación de similitud/diferencia entre ellas. La idea era entrenar un embedding (puntualmente usando la librería gensim, y el modelo [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)) para que represente en el espacio de vectores una a una las categorías, teniendo en cuenta su contexto, y luego hacer un promedio para tener la representación final de todo ese camino. Me parecía que podía sumar a la hora de elegir features que caractericen a cada publicación.

Cuando vi el atributo 'título' también se me vino a la mente el uso de embeddings, pero en este caso hay una etapa muy fuerte de limpieza (considerando faltas de ortografia, abreviaciones, uso de tildes, etc), y no iba a llegar.

Si llegaba a conseguir estas representaciones con embeddings, lo que faltaba era agregar una a una las dimensiones de los vectores como nuevas featrues para el clasificador, considerando cada dimension modela cierto aspecto.


### Training

A la hora de elegir un método de optimización para la elección de hyper parámetros, opté por *Grid Search*, pero también podría haber probado con *Random Search* y *Bayes*.

Además, podría haber agregado más configuraciones en el listado a probar.

Usé *SMOTE* para oversampling, y también podría haber probado *NearMiss* para hacer un undersampling de la clase mayoritaria.

La función de error a optimizar que elegí es f1_macro, ya que resume el presicion y recall de cada clase y al ser ['macro' sirve para problemas de desbalanceo](https://stackoverflow.com/a/53197497/9932220).


### Evaluación

Nota: más allá de que el notebook de evaluación es totalmente configurable a la hora de elegir el data set a usar, lo repliqué para que queden por un lado los resultados con el data set de test, y por el otro los de train.

Lo que esperaba ver, aparte del resultado del accuracy (no suele ser tan alto en este tipo de problemas, con tantas clases), es que al visualizar la matriz de confusión, "se marque la diagonal", y para cada clase, se vaya viendo un degrade hacia los costados; es decir, que el modelo le acierte a la gran mayoria de casos, y que si se equivoca, que sea mayoritariamente con clases cercanas. Para ilustrar un caso extremo, no estaría bueno que se confunda la clase más alta con la más baja por ejemplo, ya que no deberían parecerse.

Viendo los resulados para ambos data sets, queda en evidencia que tanto para el *XGBoost* como el *MLP Classifier*, estamos ante la presencia de *overfitting*. Los modelos no generalizan lo visto en los datos de entrenamiento, y ante datos nuevos, responden mal. Para solucionar esto, incorporaría regularización. Esto implicaría agregar nuevos valores a probar para ciertos hyperparámetros [1](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html) [2](https://medium.com/data-design/xgboost-hi-im-gamma-what-can-i-do-for-you-and-the-tuning-of-regularization-a42ea17e6ab6).

Para el caso de *Logistic Regression*, nos damos cuenta ni siquiera logró modelar la data de entrenamiento. Estamos ante un caso de *underfitting*.

Por último, para lo que es la interpretabilidad de los modelos, la librería [SHAP](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) resulta interesante a la hora de entender como funciona el modelo en general (features importance), y para un caso puntual.