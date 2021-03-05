# DSChallengeMELI

## 📌 Requerimientos
Una vez clonado el repositorio, es necesario:

- instalar [python](https://www.python.org/downloads/)
- abrir una terminal, ubicarse dentro de la carpeta **DSChallengeMELI** y correr `pip install -r requirements.txt`

## 📌 Notas

- Los notebooks asociados a cada inciso estan numerados indicando el orden en que debería correrse cada uno
- Ciertas decisiones se tomaron teniendo en cuenta:
  - poder de cómputo (16Gb RAM, procesador i5)
  - no quería demorarme demasiado en la entrega de la resolución 
 
Es por eso que se listan en éste documento algunas de las ideas que se me ocurrieron y que podriá probar en cada caso.
 
## 📌 Exploracion y analisis
En esté inciso, al traer los datos, aproveche para dejar las columnas que me interesaban para el siguiente 

En lo que respecta al armado del data set, podría haber trabajado con ...
  - información de cada producto del catálogo en sí (por ejemplo ratings, como usé los de vendedores)
  - las coordenadas de las ciudades, para agregar features de longitud/latitud, y poder representar mejor la ubicación geográfica (Ej: https://api.mercadolibre.com//classified_locations/cities/TUxBQkZMTzg5MjFa)

Además, podría haber aplicado algún método de selección de features. Lo que hice fue dejar aquellas que me parecía serían de utilidad, considerando que siempre le dedico más tiempo a ver los datos y adquirir conocimiento del dominio.

