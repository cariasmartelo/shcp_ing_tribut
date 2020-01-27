# Forecasting Fiscal Revenue in Mexico

Code of project to forecast fiscal revenue in Mexico using machine learning and econometric methods. This was produced during my 2019 summer internshp at the Mexican Ministry of Finance, at the Fiscal Income Policy Unit. The Ministry of Finance is in constant needs to forecast fiscal revenue for different time horizons. First, it is essential to estimate future cash flow and adjust accordingly. Second, the unit needs an internal approximate of the total revenue that will be achieved by end of the year. Third, the Ministry needs to submit the Federal Income Budget to the House of Representatives and forecasting revenue is a core component of it.


## Description of the Repository
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
		- Plot series. Main function to plot
		- transformation: Transform time series variables into differences, log differences, etc.
		- revert_transformation: Revert transformation to levels.
		- cross_tab: Make cross tables of variables. Mostly used to compare monthly values of two years, accumulated or in levels.
	- models_multivariate.py: Script with functions to run estimations. The main function that wraps all the sub functions is run_predictions. It takes several arguments, including the name of the model to train, the list of parameters, the initial and end dates of predictions, the DF, a list with the endogenous variables. This function receives the features, creates a DataFrame with the exogenous variables, and loops over each of the specifications in the list of parameters to run a predictive model. Depending on the name of the model that is set, it will use the functions predict_with_econometric_moder or predict_with_ml_model, which are similar in the input, in the processing of exogenous and enogenous variables and on dividing the data between training and tesitng, but different in the prediction step. While the econometric models had already a predict method included, SciKit Learning does not have one, and we needed to create our own method to do predictions recursively.
	- grid.py Auxiliary script  that defines the specifications to test, depending on the size of grid: small, medium or big.  Called by Jupyter Notebook that does predictions. 
	- dashboard.py: Script for plotly dashboard.
	- dashboard_aux.py: Auxiliary script for plotly dashboard.
JUPYTER NOTEBOOKS (.ipynb):
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



