import polars as pl
from hadlers.keywordsClassifier import keywordsClassify
from hadlers.validator import validate

def rastClassify(data, table_to, table_to_index):
    rastCls = data.getRastCls()

    if match('Subsystem', rastCls, table_to, table_to_index) > 0:
        return True
    else:
        return False

def userClassify(data, table_to, table_to_index):
    if match('Function', data.getUserCls(), table_to, table_to_index) > 0:
        return True
    else:
        return False

def kwClassify(data, table_to, table_to_index):
    for value in table_to[table_to_index, 'Function'].split('; <br>'):
        keywordsClassify(value, data)
        match('Function', data.getKwCls(), table_to, table_to_index)

def match(match_type, table_from, table_to, table_to_index):
    matchCount = 0
    for value in table_to[table_to_index, match_type].split('; <br>'):
        table_from_indexes = table_from.filter(pl.col(match_type) == value.strip())
        if len(table_from_indexes) > 0:
            for row in table_from_indexes.rows():
                addRank('Category', table_to, table_from, table_to_index, row)
                addRank('System', table_to, table_from, table_to_index, row)
                matchCount += 1
                if match_type == 'Function':
                    addRank('Subsystem', table_to, table_from, table_to_index, row)
    return matchCount

def addRank(column, t_to, t_from, row_to, row):
    value = list(row)[t_from.columns.index(column)].strip()
    if 'none' in t_to[row_to, column] and len(t_to[row_to, column].split('; <br>')) <= 1:
        t_to[row_to, column] = value
    else:
        if t_to[row_to, column]:
            column_array = t_to[row_to, column].split(';')
            column_array = [j.strip() for j in column_array]
            if value not in column_array:
                column_array.append(value)
                t_to[row_to, column] = '; '.join(sorted(column_array))

def classifyFunctions(cls_types, files, data):
    resultsList = data.getClassified()
    displayError = ''

    # Define the clean_string function
    def clean_string(value):
        if isinstance(value, str):  # Ensure the value is a string
            value = value.replace("&#39;", "'")  # Replace HTML entity for single quote
            value = value.replace("<br>", "")   # Remove <br> tags
            value = value.replace("'", "")      # Remove single quotes
            value = value.strip()               # Strip leading/trailing whitespace
        return value

    for file in files:
        error, fileContent = validate(file, "rastDownload")
        if len(error) > 0:
            print(f"Error processing file {file.filename}: {error}")  # Log the error
            displayError = error
            continue

        # Ensure 'System' and 'Category' columns exist
        if not {'System', 'Category'}.issubset(fileContent.columns):
            fileContent = fileContent.with_columns([
                pl.lit("none").alias("System"),
                pl.lit("none").alias("Category")
            ])
            fileContent = fileContent.select(['Category', 'System', 'Subsystem', 'Function'])

        # Separate rows with '- none -' in Subsystem
        fileContent_none = fileContent.filter(pl.col('Subsystem') == '- none -')
        fileContent = fileContent.filter(pl.col('Subsystem') != '- none -')

        # Add a flag column to distinguish `- none -` rows
        fileContent_none = fileContent_none.with_columns(pl.lit(True).alias("Is_None"))
        fileContent = fileContent.with_columns(pl.lit(False).alias("Is_None"))

        # Classify rows based on cls_types
        classified = False
        if "0" in cls_types:
            for index in range(len(fileContent)):
                classified = rastClassify(data, fileContent, index)

            # Clean strings in Category, System, and Subsystem after classification
            fileContent = fileContent.with_columns([
                pl.col("Category").map_elements(clean_string, return_dtype=pl.Utf8).alias("Category"),
                pl.col("System").map_elements(clean_string, return_dtype=pl.Utf8).alias("System"),
                pl.col("Subsystem").map_elements(clean_string, return_dtype=pl.Utf8).alias("Subsystem")
            ])

        elif "1" in cls_types and not classified and not data.userCls.empty:
            for index in range(len(fileContent)):
                classified = userClassify(data, fileContent, index)

        elif "2" in cls_types and not classified:
            for index in range(len(fileContent)):
                classified = kwClassify(data, fileContent, index)

        # Normalize Category, System, and Subsystem by splitting on ";" and taking the last element
        fileContent = fileContent.with_columns([
            pl.col("Category").map_elements(
                lambda x: x.split(";")[-1].strip() if isinstance(x, str) else x,
                return_dtype=pl.Utf8
            ).alias("Category"),
            pl.col("System").map_elements(
                lambda x: x.split(";")[-1].strip() if isinstance(x, str) else x,
                return_dtype=pl.Utf8
            ).alias("System"),
            pl.col("Subsystem").map_elements(
                lambda x: x.split(";")[-1].strip() if isinstance(x, str) else x,
                return_dtype=pl.Utf8
            ).alias("Subsystem")
        ])

        # Clean duplicates within the same Category
        fileContent = fileContent.group_by(['Category', 'Function']).agg(
            pl.first('System').alias('System'),
            pl.first('Subsystem').alias('Subsystem'),
            pl.first('Is_None').alias('Is_None')
        )

        # Handle duplicates across different Categories with prioritization
        fileContent = fileContent.sort(by=['Category']).with_columns(
            pl.when(pl.col('Category') == "Clustering-based subsystems")
            .then(1)
            .when(pl.col('Category') == "none")
            .then(2)
            .otherwise(0)
            .alias('Priority')
        )
        fileContent = fileContent.sort(by=['Function', 'Priority']).unique(subset=['Function'], keep='first')
        fileContent = fileContent.drop(['Priority'])
        fileContent = fileContent.drop(['Is_None'])

        # Ensure both DataFrames have the same schema before concatenation
        fileContent_none = fileContent_none.select(fileContent.columns)
        fileContent = pl.concat([fileContent, fileContent_none])

        # Add the processed DataFrame to resultsList
        if classified:
            filename = '.'.join(file.filename.split('.')[:-1])
            resultsList[filename] = fileContent

    return displayError, resultsList