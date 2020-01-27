# Forecasting Fiscal Revenue in Mexico

Code of project to forecast fiscal revenue in Mexico using machine learning and econometric methods. This was produced during my 2019 summer internshp at the Mexican Ministry of Finance, at the Fiscal Income Policy Unit. The Ministry of Finance is in constant needs to forecast fiscal revenue for different time horizons. First, it is essential to estimate future cash flow and adjust accordingly. Second, the unit needs an internal approximate of the total revenue that will be achieved by end of the year. Third, the Ministry needs to submit the Federal Income Budget to the House of Representatives and forecasting revenue is a core component of it.


## Description
Each of the folders has the following information:
- /curso python: Material that I used to teach introductory Python during Tuesdays and Thursdays.
- /inputs: Initial inputs for prediction. It has raw data downloaded initially and a subfolder
	- variables.xlsx Description of variables and download method.
	- /cuadros_preliminares: Summary statistics of tax revenue.
	- /downloads Folder with all downloads that are done using API (INEGI, Mexican Central Bank, FRED) and are updated monthly. The folder is filled up by running Jupyter Notebook download.ipynb in ../scripts/
- /Scripts: Folder with Python scripts and Jupyter Notebooks to produce analysis and execute code.
	- config.py Script with API's keys. Called by download.py.
	- download.py Script with functions to download data form API's and load to Pandas DataFrames. Script uses config.py as main source. 4 key functions: get_files_inegi(), get_files_datos_abiertos(), get_files_banxico(), and get_files_fed(). Each one downloads data form specific API. Functions to load data are many but share common structure: Read excel or csv file, y return Pandas DataFrame. Some return nominal and real values (load_ingresos_fiscales..., load_balanza_comercial...), other change the frequency of data from quartely to monthly (load_pib), and others only load indicators of INEGI,  BANXICO o de FED. Finally, two functions work to extract data from internal excel files: extract_from_cuadro_preliminar y extract_from_cuadro_isr_iva_ieps.
	- descriptive.py Main functions to describe and transform data. Las funciones principales son:
		- Plot series. Ampliamente usada para graficar en varios de los Jupyter Notebooks.
o	transformation: Para hacer transformaciones a las variables.
o	revert_transformation: Para revertir transformaciones
o	cross_tab: Para hacer tablas comparativas por año de variables. Sirve para ver comportamiento de compensaciones y devoluciones, de ISR bruto, etc.
o	cross_tab_lif: Similar a la anterior, pero especifica para hacer comparaciones con la Ley de Ingresos.
•	models_multivariate.py: Script con las funciones para hacer estimaciones. La función que arropa todo el script es run_predictions. Esta función toma varios argumentos, como el nombre del modelo a estimar, la lista de parámetros, los inicios y finales de las predicciones, el DF, una lista con las columnas endógenas, etc. Esta función, recibe los datos, crea un DataFrame con las variables exógenas dependiendo de los parámetros que el usuario haya incluido, y después hace un loop para cada una de las especificaciones en la lista de parámetros, correr un modelo predictivo. Dependiendo del nombre del modelo, usa a las funciones predict_with_econometric_model o predict_with_ml_model. Las funciones predict_with_econometric_model y predict_with_ml_model son similares en que recibe los datos, crean un DataFrame con los datos endógenos y exógenos, dividen los DataFrames en train y en test usando la función split_train_test, dividen los parámetros del modelo entre parámetros que se deben pasar al llamar .fit() o al construir el modelo y obtienen las predicciones llamando la función run_model. Se diferencian en un aspecto clave: Mientras los métodos econométricos ya contaban con un método que hacia las predicciones paso a paso y de manera recursiva, los métodos de machine learning no, y hubo que implementar estos pasos. 
•	grid.py Script que sirve para construir el grid completo de posibles parámetros de cada modelo. Es llamado por el Jupyter Notebook de predicciones. Tiene tres tamaños de grid: small, medium y big.
•	dashboard.py: Código para crear dashboard usando dash y plotly. El código que tiene los botones interactivos y las gráficas interactivas del dashboard que hemos visto.
•	dashboard_aux.py: Script auxiliar del dashboard. Tiene las listas de variables necesarias para crear el dashboard.
JUPYTER NOTEBOOKS (Terminación .ipynb):
•	download.ipynb: Jupyter notebook para hacer descarga de datos de INEGI, BANXICO, FED y Datos abiertos  (Estadísticas oportunas)
•	Estimacion_marco_macro.ipynb: Jupyter notebook con estimaciones del marco macro mexicano y de Estados Unidos.
•	Estimaciones_cierre.ipybn: Jupyter notebook con estimaciones finales de ingresos tributarios. Usa modelos seleccionados por Pipeline_analyze.ipynb
•	Machote_eficiencia_recaudatoria.ipynb: Jupyter notebook Machote para hacer análisis de eficiencia recaudatoria, así como cambio anual en ingresos tributarios, cambio en compensaciones y predecir el PIB.
•	Pipeline_descriptive.ipynb: Jupyter notebook con descripción de variables recaudatorias, análisis de estacionariedad y descomposición en ciclo y tendencia.
•	Pipeline_multivariate_descriptive.ipynb: Jupyter notebook con análisis descriptivo de recaudación y otras variables. Se analiza comportamiento de variables Macro de USA y de México.
•	Pipeline_univariate_predict.ipynb Jupyter notebook para hacer predicciones de ingresos tributarios sin covariables. Fueron las primeras estimaciones hechas. No son las estimaciones usadas en el reporte final.
•	Pipeline_multivariate_predict.ipynb: Jupyter notebook donde se corren todas las estimaciones. Este es el notebook que utiliza el script models_multivariate.py de manera importante, y estima todos los modelos para cada uno de los cortes transversales. En este notebook se producen los resultados.
•	Pipeline_analyze.ipynb Jupyter notebook donde se leen los csv de resultados y se obtienen los mejores modelos. Se hacen gráficas de mejores modelos y se seleccionan los modelos que después se utilizan en estimaciones de cierre.
 	SUBCARPETAS
•	/reportes_eficiencia: Carpeta con reporte de eficiencia recaudatoria (Versiones estáticas del Machote de Eficiencia)
•	/assets: Carpeta con imágenes del dashboard
•	/estimaciones: Estimaciones de cierre de los ingresos tributarios
/figures Gráficas generadas en el análisis de recaudación. Tiene gráficas que resultaron de Machote_multivariado.ipybn
/Presentaciones eficiencia Presentaciones de eficiencia recaudatoria
/Presentaciones_prediccion Presentaciones del proyecto de predicción. Incluye la presentación de avance con el BID, presentación inicial con Víctor Higo y presentación final.
/reporte Carpeta con reporte final. Incluye archivo LaTeX (.tex), archivos intermedios (.aux, .log y .out) y archivo pdf, así como carpeta con imágenes.
/results Carpeta con resultados de las estimaciones realizadas. Son archivos .csv que después son importados en el Jupyter Notebook Pipeline_analyze.ipynb
•	Calendario_lif.pdf Calendario de la Ley de Ingresos de 2019
•	README.md archivo de texto similar a este para si presentación en GitHub.
•	Cronograma.xlsx Aproximación del cronograma seguido en el proyecto.
•	Requirementes.txt: Liberias que se requieren para correr todas las funciones



