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
	- dashboard_aux.py: Auxiliary script for plotly dashboard.\\
JUPYTER NOTEBOOKS (.ipynb):
	- download.ipynb: Jupyter notebook to download data from INEGI, BANXICO, FED y Datos abiertos using API
	- Estimacion_marco_macro.ipynb: Jupyter notebook with forecasts of Mexican and US Macroeconomic variables.
	- Estimaciones_cierre.ipybn: Jupyter notebook con with final estimations of tax revenue. Uses models selected by Pipeline_analyze.ipynb
	- Machote_eficiencia_recaudatoria.ipynb: Jupyter notebook to do monthly analysis of tax revenue and forecast GDP.
	- Pipeline_descriptive.ipynb: Jupyter notebook con visualization of tax revenue, estationarity analysis and tend decomposition of variables.
	- Pipeline_multivariate_descriptive.ipynb: Jupyter notebook with descriptive analysis of how multiple macroeconomic variables relate.
	- Pipeline_univariate_predict.ipynb Jupyter notebook to predict fiscal revenue without other enogenous variables.
	- Pipeline_multivariate_predict.ipynb: Jupyter notebook to run all moels. Heavily relies on models_multivariate.py and estimates all models for each of the time horizons. This notebook outputs main results. \
	- Pipeline_analyze.ipynb Jupyter notebook to a analyze results.
### Subfolders
	- /reportes_eficiencia: Reports with analysis of efficiency of tax collection
	- /assets: Dashboard images
	- /estimaciones: Final estimations of tax revenue.
	- /figures Graphs that describe tax revenue. Many created in Machote_multivariado.ipybn
- /Presentaciones eficiencia Power point presentations of tax collection efficiency.
- /Presentaciones_prediccion Power point presentations of the whole project.
- /reporte Final repor. Includes LaTeX (.tex) file, other files (.aux, .log y .out) and pdf, as well as images.
- /results Results of estimations. csv files imported by Pipeline_analyze.ipynb
- Calendario_lif.pdf Calendar of Federal Income Budget of 2019.
- Requirementes.txt: Required Python libraries to run analysis.



