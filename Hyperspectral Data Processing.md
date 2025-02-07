Fuente: [Hyperspectral Data Processing](https://books.google.es/books?hl=es&lr=&id=uP1gRRfkMxgC&oi=fnd&pg=PR23&dq=hyperspectral+signature&ots=K04o3PvI3Z&sig=2oVj3E6jKhg69eoWqQMQ6pjXy5E#v=onepage&q&f=false)
# Data Dimensionality Reduction

La **reducción de la dimensionalidad (RD)** se ha utilizado en la explotación de datos hiperespectrales con diversos fines. En particular, se ha utilizado como técnica de preprocesamiento para reducir un espacio de datos de muy alta dimensionalidad a un espacio manejable de baja dimensionalidad en el que el análisis de datos pueda realizarse con mayor eficacia. Hay dos enfoques comunes que se utilizan ampliamente para la DR:
<ol>

<li><strong>DR por transformación (DRT)</strong>. Todas estas técnicas de transformación pueden ser muy útiles en el tratamiento de imágenes hiperespectrales.</li>
<ol>
<li><strong>Análisis de Componentes (CA)</strong>. Una transformación CA se considera generalmente como una transformación que utiliza la estadística como criterio para descorrelacionar y convertir los datos en un conjunto de componentes de datos no correlacionados para su análisis.</li>
<ul>
<li><strong>Análisis de componentes principales (PCA)</strong>.</li>
<li>Fracción de ruido máxima (MNF) basada en la relación señal/ruido (SNR)</li>
</ul>
<li><strong>Extracción de Características (FE)</strong>. Una transformación FE utiliza un criterio basado en la extracción de características para producir un conjunto de vectores de características que permitan representar los datos.
</li>
<ul><li><strong>Análisis discriminante lineal basado en la proporción de Fisher (FLDA)</strong>.</li></ul>
</ol>

<li><strong>DR por selección de banda (DRBS)</strong> busca un subconjunto de bandas que represente los datos originales, de modo que la información de interés de los datos pueda conservarse en el subconjunto de bandas seleccionado. Curiosamente, como se verá, la mayoría de los criterios diseñados para la DRT también son aplicables a la DRBS.</li>
</ol>
## Análisis de componentes

### 1. Análisis de Componentes Principales (PCA)

PCA es una técnica estadística que transforma los datos en un nuevo sistema de coordenadas donde las variables (componentes principales) están ordenadas según la cantidad de varianza que existe en los datos. Su objetivo principal es la reducción de dimensionalidad, eliminando redundancias y manteniendo la mayor cantidad de información posible. PCA se basa en la matriz de covarianza de los datos sin necesidad de conocer la distribución de probabilidad.
#### Código
``` c++
#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Función para convertir la imagen en una matriz de datos
MatrixXf convertirImagenAMatriz(const vector<vector<vector<float>>>& imagen) {
    int filas = imagen.size();
    int columnas = imagen[0].size();
    int bandas = imagen[0][0].size();
    MatrixXf datos(filas * columnas, bandas);

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            for (int k = 0; k < bandas; k++) {
                datos(i * columnas + j, k) = imagen[i][j][k];
            }
        }
    }
    return datos;
}

// Función para realizar PCA y reducir la dimensionalidad
MatrixXf aplicarPCA(const MatrixXf& datos, int num_componentes) {
    VectorXf media = datos.colwise().mean();
    MatrixXf datosCentrados = datos.rowwise() - media.transpose();

    // Calcular matriz de covarianza
    MatrixXf covarianza = (datosCentrados.adjoint() * datosCentrados) / float(datos.rows() - 1);

    // Descomposición en valores propios
    SelfAdjointEigenSolver<MatrixXf> solver(covarianza);
    MatrixXf autovectores = solver.eigenvectors().rightCols(num_componentes);

    // Proyectar los datos en los componentes principales
    return datosCentrados * autovectores;
}

int main() {
    // Supongamos una imagen 4x4 con 5 bandas
    vector<vector<vector<float>>> imagen(4, vector<vector<float>>(4, vector<float>(5, 1.0)));

    // Convertir a matriz
    MatrixXf datos = convertirImagenAMatriz(imagen);

    // Aplicar PCA para reducir a 3 componentes principales
    MatrixXf datosReducidos = aplicarPCA(datos, 3);

    cout << "Datos reducidos: \n" << datosReducidos << endl;

    return 0;
}

```
### 2. Transformada de Karhunen-Loève (KLT)

KLT es un método más general que el PCA, utilizado en el procesamiento de señales. Se basa en la descomposición de una señal en funciones ortogonales llamadas **eigenfunciones** en términos de error cuadrático medio (MSE). KLT funciona en un dominio de tiempo continuo, mientras que PCA es su versión discreta aplicada a datos representados en matrices.
### 3. Expansión de Karhunen-Loève (KL Expansion)

Es una forma más generalizada de la KLT utilizada en el análisis de señales estadísticas, donde se descompone una señal en una serie de funciones ortogonales (eigenfunciones). Se diferencia de PCA en que requiere información sobre la distribución de probabilidad de los datos.

En resumen, PCA es una versión discreta de KLT utilizada cuando los datos están en forma de matrices, mientras que KLT y KL Expansion operan en dominios más generales y con supuestos estadísticos más fuertes.
## Extracción de características
### 1. Análisis Discriminante Lineal de Fisher (FLDA)

El **Análisis Discriminante Lineal de Fisher (FLDA)** es una técnica de reducción de dimensionalidad que busca encontrar una proyección óptima de los datos para maximizar la separabilidad entre clases. Es útil en problemas de clasificación y se basa en la relación entre las varianzas dentro de cada clase y entre clases.

1. Problema de PCA
	- <strong>PCA</strong> encuentra direcciones que maximizan la varianza de los datos, pero <strong>no necesariamente mejora la separabilidad de clases</strong>.
	- Por eso, FLDA se usa cuando queremos <strong>maximizar la separación entre diferentes clases</strong>.

2. Concepto clave:
	- FLDA busca una transformación lineal que maximice la **razón de Fisher**: $$ J(w) = \frac{\text{Varianza entre Clases}}{\text{Varianza dentro de las Clases}}$$
	- Esto significa que proyectamos los datos en una nueva dirección que separa mejor las clases.

3. Matriz de dispersión entre clases ($S_B$) y dentro de las clases ($S_W$​)
	- $S_B$ mide la dispersión entre las medias de las clases.
	- $S_W$ mide la dispersión dentro de cada clase.
	- FLDA busca maximizar la relación $S_B$ /$S_W$​  para encontrar la mejor proyección.

4. Proceso en FLDA
	- Se calculan los vectores de características que maximizan la razón de Fisher.
	- Para ***p* clases**, se generan ***p* - 1 fronteras de decisión**.
	- Esto reduce la dimensionalidad del problema a ***p* - 1**.
#### Código 
1. Convertir la imagen hiperespectral a una matriz de píxeles vs. bandas.
2. Calcular las medias de cada clase y la media global.
3. Calcular las matrices de dispersión entre clases $S_B$ y dentro de las clases $S_W$​
4. Resolver el problema de valores propios para encontrar los vectores discriminantes.
5. Proyectar la imagen a un espacio de menor dimensión usando los vectores discriminantes.
```c++
#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Definir una imagen hiperespectral: [filas][columnas][bandas]
using ImagenHiperespectral = vector<vector<vector<float>>>;

// Estructura para almacenar datos de cada clase
struct Clase {
    vector<VectorXf> muestras;
    VectorXf media;
};

// Convertir imagen hiperespectral en una matriz 2D [pixeles x bandas]
vector<VectorXf> convertirImagenAMatriz(const ImagenHiperespectral &imagen) {
    int filas = imagen.size();
    int columnas = imagen[0].size();
    int bandas = imagen[0][0].size();
    
    vector<VectorXf> matriz;
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            VectorXf pixel(bandas);
            for (int b = 0; b < bandas; b++) {
                pixel(b) = imagen[i][j][b];
            }
            matriz.push_back(pixel);
        }
    }
    return matriz;
}

// Calcular la media de cada clase
VectorXf calcularMedia(const vector<VectorXf> &muestras) {
    int n = muestras.size();
    VectorXf media = VectorXf::Zero(muestras[0].size());

    for (const auto &m : muestras) {
        media += m;
    }
    return media / n;
}

// Aplicar FLDA a las clases
MatrixXf aplicarFLDA(const vector<Clase> &clases, int componentes) {
    int dimension = clases[0].media.size();
    int numClases = clases.size();

    // Calcular la media global
    VectorXf mediaGlobal = VectorXf::Zero(dimension);
    int totalMuestras = 0;
    for (const auto &c : clases) {
        mediaGlobal += c.media * c.muestras.size();
        totalMuestras += c.muestras.size();
    }
    mediaGlobal /= totalMuestras;

    // Matrices de dispersión dentro de clases (Sw) y entre clases (Sb)
    MatrixXf Sw = MatrixXf::Zero(dimension, dimension);
    MatrixXf Sb = MatrixXf::Zero(dimension, dimension);

    for (const auto &c : clases) {
        MatrixXf Si = MatrixXf::Zero(dimension, dimension);
        for (const auto &x : c.muestras) {
            VectorXf diff = x - c.media;
            Si += diff * diff.transpose();
        }
        Sw += Si;

        VectorXf diffMedia = c.media - mediaGlobal;
        Sb += c.muestras.size() * (diffMedia * diffMedia.transpose());
    }

    // Resolver el problema de valores propios para Sw^-1 * Sb
    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXf> solver(Sb, Sw);
   MatrixXf W = solver.eigenvectors().rightCols(componentes); // Tomamos los autovectores más importantes

    return W;
}

// Proyectar la imagen en el nuevo espacio discriminante
vector<VectorXf> proyectarDatos(const vector<VectorXf> &datos, const MatrixXf &W) {
    vector<VectorXf> proyectados;
    for (const auto &x : datos) {
        proyectados.push_back(W.transpose() * x);
    }
    return proyectados;
}

int main() {
    // Crear una imagen hiperespectral de ejemplo (4x4 píxeles con 5 bandas)
    ImagenHiperespectral imagen(4, vector<vector<float>>(4, vector<float>(5)));

    // Llenar con valores aleatorios simulando datos espectrales
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int b = 0; b < 5; b++) {
                imagen[i][j][b] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }

    // Convertir imagen a matriz 2D
    vector<VectorXf> datos = convertirImagenAMatriz(imagen);

    // Definir clases manualmente (en la práctica se etiquetan los píxeles)
    Clase clase1, clase2;
    for (size_t i = 0; i < datos.size() / 2; i++) {
        clase1.muestras.push_back(datos[i]);
    }
    for (size_t i = datos.size() / 2; i < datos.size(); i++) {
        clase2.muestras.push_back(datos[i]);
    }
    
    // Calcular la media de cada clase
    clase1.media = calcularMedia(clase1.muestras);
    clase2.media = calcularMedia(clase2.muestras);

    // Aplicar FLDA y reducir la imagen a 2 componentes discriminantes
    vector<Clase> clases = {clase1, clase2};
    MatrixXf W = aplicarFLDA(clases, 2);

    // Proyectar los datos en el nuevo espacio reducido
    vector<VectorXf> datosReducidos = proyectarDatos(datos, W);

    // Mostrar los primeros datos transformados
    cout << "Datos proyectados en el espacio discriminante:" << endl;
    for (size_t i = 0; i < min(datosReducidos.size(), size_t(5)); i++) {
        cout << datosReducidos[i].transpose() << endl;
    }

    return 0;
}
```

### 2. Proyección Ortogonal de Subespacios (OSP)

La Proyección Ortogonal de Subespacios (OSP) es una técnica que se utiliza en el procesamiento de imágenes hiperespectrales para separar la información útil de la interferencia o el ruido, mediante la proyección de los datos en un subespacio que es ortogonal (perpendicular) a ciertas componentes indeseadas.

1. **Modelado de la Señal Hiperspectral**
	En una imagen hiperespectral, cada píxel se compone de un vector espectral con un número elevado de bandas. Se suele modelar cada píxel *r* como una combinación lineal de unas firmas espectrales puras (endmembers) más un componente de ruido:$$ r = Mα + n $$
	- $M$ es la matriz de endmembers (de tamaño $\text{número de bandas} \times p$) en la que cada columna representa la firma espectral de un material puro.
	- $α$ es el vector de coeficientes de abundancia, que indica en qué proporción contribuye cada endmember al píxel.
	- $n$ es el ruido o error en la medición.

2. **Identificación del Subespacio de Interés**
	El conjunto de endmembers define un **subespacio** dentro del espacio espectral original. Por ejemplo, si existen *p* endmembers, en teoría los datos relevantes (la parte “útil” de la señal) se concentrarán en un subespacio de dimensión, a lo sumo, *p* (o incluso menor, si existe redundancia).
	
	En muchas aplicaciones, se dispone de información o se pueden aplicar métodos (como N-FINDR, VCA, etc.) para estimar estos endmembers.

3. **Construcción de la Matriz de Proyección**
	El objetivo de la OSP es “eliminar” la influencia de ciertos componentes (por ejemplo, aquellos correspondientes a materiales conocidos o al fondo) o, en otros casos, aislar la señal de interés. Para ello, se construye una **matriz de proyección** que proyecta los datos en el **complemento ortogonal** del subespacio definido por *M*.
	
	La forma clásica de obtener esta proyección es: $$P=I-M(M^TM)^{-1}M^T$$
	donde:
	- $I$ es la matriz identidad (del tamaño de la dimensión espectral).
	- $M(M^TM)^{-1}M^T$ es la proyección de cualquier vector sobre el subespacio generado por las columnas de $M$.
	Con $P$, lo que hacemos es “restar” la parte que se encuentra en el subespacio definido por $M$; es decir, $P$ proyecta cualquier vector en el **complemento ortogonal** a ese subespacio.

4. **Aplicación de la Proyección a los Datos**
	Una vez obtenida la matriz $P$, cada píxel $r$ se transforma de la siguiente manera: $$r'=Pr$$
	El vector $r'$ resultante contiene la componente del píxel que es **ortogonal** al subespacio de $M$. Esto significa que, si $M$ representa las firmas de ciertos materiales (por ejemplo, el fondo o materiales que se desean eliminar), $r′$ tendrá reducida o eliminada esa contribución, resaltando la información que se encuentra fuera de ese subespacio.
	
	Esta proyección es especialmente útil en tareas como:
	- **Detección de objetivos:** Se suprime el fondo (o la mezcla indeseada) para facilitar la detección de un material o target específico.
	- **Reducción de dimensionalidad**: Al enfocarnos en la parte de la señal que realmente varía de forma independiente de los endmembers conocidos, se puede trabajar con menos dimensiones sin perder la información de interés.

5. **Interpretación y Consideraciones**
	- **Eliminación de Interferencias:**  Al proyectar sobre el complemento ortogonal del subespacio definido por $M$, se eliminan o se reducen las componentes que se alinean con los endmembers. Esto es útil para suprimir interferencias o para eliminar la influencia del fondo.
	
	- **Detección y Clasificación:** Una vez realizada la proyección, la señal resultante $r'$ puede ser más fácil de clasificar o de analizar, ya que se ha filtrado la información redundante o indeseada.
	
	- **Dependencia de la Matriz $M$:**  La efectividad de la OSP depende en gran medida de la correcta identificación de los endmembers. Si $M$ no captura adecuadamente la estructura de los materiales presentes, la proyección puede no eliminar la interferencia de forma efectiva.
	
	- **Estabilidad Numérica**. El cálculo de $(M^TM)^{-1}$ requiere que las columnas de $M$ sean linealmente independientes. En la práctica, puede ser necesario aplicar técnicas de regularización o usar métodos numéricamente estables para invertir la matriz.
#### 2.1 Relación entre OSP y Clustering
El clustering en imágenes hiperespectrales consiste en **agrupar píxeles con firmas espectrales similares**, asumiendo que corresponden a los mismos materiales. Sin embargo, debido a la mezcla espectral y al ruido, separar estos grupos puede ser complicado.

Aquí es donde OSP ayuda:
- **Filtra interferencias**: Al proyectar sobre el complemento ortogonal de endmembers conocidos, OSP reduce la influencia de materiales que no son de interés.
- **Mejora la separabilidad de clases**: Los píxeles que pertenecen a distintos materiales se proyectan en diferentes direcciones, lo que puede hacer que los clusters sean más distinguibles.
- **Reduce la dimensionalidad**: Al trabajar en un subespacio relevante, los algoritmos de clustering pueden operar más eficientemente.
#### 2.2 Flujo de Trabajo: OSP + Clustering

1.  Preprocesamiento de la Imagen Hiperespectral
	- Representar la imagen como una matriz $X$ de tamaño $\text{número de bandas} \times \text{número de píxeles}$.
	- Opcionalmente, realizar **normalización o reducción de ruido** (por ejemplo, usando PCA).

2. Determinación de los Endmembers
	- Utilizar un método como **N-FINDR, VCA (Vertex Component Analysis)** o **PPI (Pixel Purity Index)** para identificar los endmembers espectrales dominantes en la imagen.
	- Esto proporciona la matriz $M$, que representa las firmas espectrales de los materiales puros.

3. Aplicación de OSP
	- Construir la **matriz de proyección**: $P=I-M(M^TM)^{-1}M^T$
	- Proyectar cada píxel $r$ usando $r′=Pr$, de modo que la nueva representación resalte la información no contenida en los endmembers de $M$.

4. Aplicación de Clustering
	Una vez que los píxeles han sido transformados por OSP, se pueden aplicar técnicas de clustering como:
	- **K-Means**: Para agrupar píxeles en $k$ clusters según su firma espectral filtrada.
	- **DBSCAN**: Para encontrar agrupaciones densas y detectar anomalías.
	- **GMM (Gaussian Mixture Models)**: Para modelar la distribución espectral de los píxeles y obtener probabilidades de pertenencia a cada clase.
