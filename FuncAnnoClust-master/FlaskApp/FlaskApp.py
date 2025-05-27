import csv
import json
from flask import Flask, request, render_template, send_file, make_response, Response, jsonify, Response, stream_with_context
import pandas as pd
import polars as pl
from hadlers.Data import Data
import xlsxwriter
from hadlers.classifier import classifyFunctions
from hadlers.AllClasDownload import all_classified
from hadlers.clsDisplay import displayClassification
from hadlers.counter import countFunctions
from hadlers.visualize_statistics import buildPlots
from hadlers.visualize_statistics import statistic_test
from hadlers.validator import validate
from hadlers.validator import is_int, is_float
from hadlers.memoryzip_plots import get_zip_buffer
from sys import getsizeof
from typing import List
import os
import io
import logging


logging.basicConfig(level=logging.DEBUG)

from logging.config import dictConfig

import settings
from typing import List

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


app = Flask(__name__)
data = Data()
allowed_file_types = ['csv']

if data.getRastCls().shape[0] == 0:
    path_to_rast_classification = os.path.join(settings.ROOT_DIR, 'static/csvFiles/rastClassification.csv')
    data.setRastCls(pl.read_csv(path_to_rast_classification))

if getsizeof(data.getHierarchy):
    hierarchy = displayClassification(data)
    data.setHierarchy(hierarchy)

@app.route('/')
def index():
     return render_template("index.html")

@app.route('/analisis', methods=['get', 'post'])
def analyse():
    return render_template("analisis.html", dict=data.getClassified(), hierarchy=data.getHierarchy(), countTable=data.getCount(), plots=data.getPlots(), displayCount=1)

@app.route('/documentation', methods=['get'])
def documentation():
    return render_template("documentation.html")

# можно убрать
@app.route('/reset', methods=['post'])
def reset():
    data.reset()
    return render_template("analisisCls.html", dict=data.getClassified(), displayCount=1)


@app.route('/classify', methods=['post'])
def classify():
    # data.reset()
    cls_types = request.form.getlist('cls_type')
    rastDownloads = request.files.getlist("rastDownloads[]")
    userCls = request.files.get("userCls")

    errors = []
    if (userCls):
        errorUserCls, validated = validate(userCls, "userCls")
        data.setUserCls(validated)
        errors.append(errorUserCls)

    errorDownloads, validated = classifyFunctions(cls_types, rastDownloads, data)
    data.setClassified(validated)  
    errors.append(errorDownloads)

    return render_template("analisisCls.html", dict=data.getClassified(), displayCount=1, errors=errors)

@app.route('/fullClassified', methods=['get', 'post'])
def fullClassified():
    dict = data.getClassified()
    return render_template("fullClassified.html", dict=dict, displayCount=len(dict))


@app.route('/count', methods=['post'])
def count():
    data.setResCount()
    data.resComputedMatrix()
    request_json_data = request.get_json()
    data.setCount(countFunctions(data, request_json_data))

    return render_template("analisisCount.html", countTable=data.getCount())


@app.route('/visualize', methods=['POST'])
def visualize():
    # Initialize default values
    random_state = 0
    dbscan_eps = 0.05
    optics_max_eps = 0.5
    damping = 0.9  # Default damping for Affinity Propagation
    errors = []

    # Retrieve form inputs
    try:
        if request.form.get('bayesian_gaussian_mixture__input'):
            random_state = int(request.form.get('bayesian_gaussian_mixture__input'))

        if request.form.get("DBSCAN__input"):
            dbscan_eps = float(request.form.get("DBSCAN__input"))

        if request.form.get("optics__eps_input"):
            optics_max_eps = float(request.form.get("optics__eps_input"))

        if request.form.get("affinity_propagation__damping_input"):
            damping = float(request.form.get("affinity_propagation__damping_input"))

    except ValueError as e:
        errors.append(f"Неверный формат ввода: {e}")

    # Validate random_state
    if not isinstance(random_state, int):
        errors.append("Значение random_state должно быть целым числом.")

    # Validate perplexity
    try:
        perplexity_numb = float(request.form['perplexity'])
        strains_numb = len(data.getCount())
        if perplexity_numb >= strains_numb:
            errors.append(f"Значение perplexity должно быть меньше количества штаммов: {strains_numb}.")
        elif perplexity_numb <= 0:
            errors.append("Значение perplexity должно быть больше 0.")
    except ValueError:
        errors.append("Значение perplexity должно быть числом.")

    # Validate DBSCAN eps
    if dbscan_eps is not None and (not isinstance(dbscan_eps, float) or dbscan_eps <= 0):
        errors.append("Значение eps для DBSCAN должно быть положительным числом.")

    # Validate OPTICS max_eps
    if optics_max_eps is not None and (not isinstance(optics_max_eps, float) or optics_max_eps <= 0):
        errors.append("Значение max_eps для OPTICS должно быть положительным числом.")

    # Validate Affinity Propagation damping
    if not (0.5 <= damping < 1.0):
        errors.append("Значение damping для Affinity Propagation должно быть между 0.5 и 1.0.")

    # Check for errors before proceeding
    if len(errors) == 0:
        # Prepare parameters for buildPlots
        params = dict(
            data=data,
            methods=request.form.getlist('method'),  # Visualization methods (e.g., PCA, t-SNE)
            perplexity=request.form['perplexity'],
            clusterMethods=request.form.getlist('clusterMethod'),  # Clustering methods
            n_clusters=request.form['n_clusters'],  # Number of clusters
            linkage=request.form['linkage'],  # Linkage for hierarchical clustering
            distance_metric=request.form["convergenceType"],  # Distance metric
            random_state=random_state,
            eps=dbscan_eps,
            optics_max_eps=optics_max_eps,
            damping=damping,
            spectral_affinity=request.form.get("spectral_clustering__affinity_select", "nearest_neighbors")
        )

        # Generate plots
        data.setPlots(buildPlots(**params))

    # Return the template with plots and error messages
    return render_template("analisisVsl.html", plots=data.getPlots(), errors=errors)

@app.route('/download/<type>/<filename>', methods=['GET'])
def download(type, filename):
    if type == "classified":
        df = data.getClassified()[filename].write_csv(separator=";")
    if type == "kwClassification":
        df = data.getKwCls().write_csv(separator=";")
    if type == "counted":
        df = data.getCount().write_csv(separator=";")
    return Response(df,status=200,headers={"Content-disposition":"attachment; filename="+filename+".csv"}, mimetype="application/csv")


@app.route('/download/plots', methods=['GET'])
def download_plots():
    try:
        export_format = request.args.get('export_format')
        logging.debug(f"Export format: {export_format}")
        if not export_format or export_format not in ["png", "jpeg", "svg", "pdf"]:
            return "Error: Invalid or missing 'export_format' parameter.", 400

        buff = get_zip_buffer(data, export_format)
        return send_file(buff, mimetype='application/zip', as_attachment=True, download_name="all_plots.zip")
    except Exception as e:
        logging.error(f"Error in download_plots: {e}")
        return f"Error: {str(e)}", 500



@app.route('/download/all_classified', methods=['GET'])
def download_allclass():
    try:
        # Call the function to get the classified DataFrame (assumed to be a Polars DataFrame)
        classified_df = all_classified(data)

        # Check if the DataFrame is None or empty
        if classified_df is None or classified_df.height == 0:
            logging.warning("No classifications available.")
            return Response(
                "Error: No classifications available.",
                status=404,
                mimetype="text/plain"
            )

        # Define the filename and headers for the response
        filename = "classified_all_data.xlsx"
        headers = {
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }

        # Convert the Polars DataFrame to a Pandas DataFrame
        pandas_df = classified_df.to_pandas()

        # Create an in-memory buffer to store the Excel file
        output_buffer = io.BytesIO()

        # Write the Pandas DataFrame to the buffer using ExcelWriter
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            # Explicitly write the DataFrame to a sheet named "ClassifiedData"
            pandas_df.to_excel(writer, sheet_name="ClassifiedData", index=False)

        # Reset the buffer's position to the beginning
        output_buffer.seek(0)

        # Return the file as a streamed response
        return Response(
            output_buffer.getvalue(),
            status=200,
            headers=headers
        )

    except Exception as e:
        # Log the exception for debugging
        logging.error(f"An error occurred while downloading cluster data: {str(e)}")
        return Response(
            f"Error: {str(e)}",
            status=500,
            mimetype="text/plain"
        )

@app.route('/download/cluster', methods=['GET'])
def download_cluster():
    try:
        # Retrieve cluster predictions and test statistics
        predictions_df = data.getCluster()
        test_stat = data.getStatResults()

        # Check if predictions_df is None
        if predictions_df is None and test_stat is None:
            logging.warning("No cluster predictions and statitic tests available.")
            return Response(
                "Error: No cluster predictions available.",
                status=404,
                mimetype="text/plain"
            )

        # Transform test_stat into a list of rows
        test_stat_rows = []

        for method_name, test_data in test_stat.items():
            if len(test_data) < 6:
                logging.error(f"Insufficient data for test '{method_name}'. Expected at least 6 elements in the array.")
                return Response(
                    f"Error: Insufficient data for test '{method_name}'. Expected at least 6 elements in the array.",
                    status=500,
                    mimetype="text/plain"
                )
            row = {
                "Method name": method_name,
                "Test statistic name": test_data[1],  # Convert to string for safety
                "Sample size": test_data[2],         # Convert to integer
                "Number of groups": test_data[3],    # Convert to integer
                "Test statistic": test_data[4],    # Convert to float
                "P-Value": test_data[5],           # Convert to float
                "Number of Permutations": test_data[6]  # Convert to integer
            }
            test_stat_rows.append(row)

        # Convert the list of rows into a DataFrame
        test_stat_df = pd.DataFrame(test_stat_rows)

        # Define constants for headers and filename
        FILENAME = "cluster_data.xlsx"
        HEADERS = {
            "Content-Disposition": f"attachment; filename={FILENAME}",
            "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }

        # Create an in-memory buffer to store the Excel file
        output = io.BytesIO()

        # Use pandas.ExcelWriter to write multiple sheets
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # Write predictions_df to the first sheet
            predictions_df.to_excel(writer, sheet_name="Cluster_Predictions", index=True)

            # Write test_stat_df to a single sheet
            test_stat_df.to_excel(writer, sheet_name="Test_Results", index=False)

        # Seek to the beginning of the buffer
        output.seek(0)

        # Stream the Excel content to avoid high memory usage
        def generate_excel():
            yield output.getvalue()

        return Response(
            stream_with_context(generate_excel()),
            status=200,
            headers=HEADERS
        )

    except Exception as e:
        # Log the exception for debugging
        logging.error(f"An error occurred while downloading cluster data: {str(e)}")
        return Response(
            f"Error: {str(e)}",
            status=500,
            mimetype="text/plain"
        )

@app.route('/uploadBreakdown', methods=['post'])
def uploadBreakdown():
    breakdown = request.files.get("breakdown")
    error, validated = validate(breakdown, 'breakdown')
    data.setBreakdown(validated)
    return  render_template("breakdown.html", df=data.getBreakdown(), error=error)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Проведение оценки статистической достоверности различий на странице анализа"""
    random_state = 0
    dbscan_eps = 0.05
    if bool(request.form.get('bayesian_gaussian_mixture__input')):
        random_state = request.form.get('bayesian_gaussian_mixture__input')
    if bool(request.form.get("DBSCAN__input")):
        dbscan_eps = request.form.get("DBSCAN__input")

    # distance_metric = request.form["convergenceType"]
    clusterMethods = []
    errors=[]

    if not is_int(random_state):
        errors.append("Значение random_state должно быть целым")

    request_cluster = request.form.getlist('clusterMethod')

    if len(request_cluster) and "none" not in request_cluster:
        print(request.form.getlist('clusterMethod'))
        clusterMethods = request_cluster
    else:
        errors.append("Необходимо выбрать тип кластеризации")


    if len(errors) < 1:
        params = dict(
            data=data,
            statMethods=request.form.getlist('statMethod'),
            clusterMethods=clusterMethods,
            distance_metric=request.form["convergenceType"],
            n_clusters=request.form['n_clusters'],
            linkage=request.form['linkage'],
            # tree=tree,
            # otu_ids=otu_ids,
            random_state=random_state,
            eps=dbscan_eps
        )

        errors, stat_result = statistic_test(**params)
        data.setStatResults(stat_result)

    errors: List[str] = errors  # Сюда передать список из ошибок

    return render_template('statistic_test.html', result=data.getStatResults(), errors=errors)


# if __name__ == "__main__":
#     app.run(debug=settings.DEBUG)

if __name__ == '__main__':
    app.run(host='0.0.0.0')