import polars as pl
from skbio import TreeNode
from io import StringIO
import re

def template(allowed_file_types, data, columns, error):
    content = pl.DataFrame()
    file_type = data.filename.split('.')[-1]
    if file_type in allowed_file_types:
        if file_type =='tsv':
            content = pl.read_csv(data, separator='\t')
        else:
            content = pl.read_csv(data, separator=';')

        if set(columns).issubset(content.columns):
            error = ''
        else:
            content = pl.DataFrame()
    return error, content

def validate(data, type):
    if type == "userCls":
        allowed_file_types = ['csv']
        columns = ['Function','Subsystem', 'System', 'Category']
        error = 'Неверный формат пользовательской классификации'

    if type == "rastDownload":
        allowed_file_types = ['csv', 'tsv']
        columns = ['Function','Subsystem']
        error = 'Неверный формат выгрузок из RAST'

    if type == "breakdown":
        allowed_file_types = ['csv']
        columns = ['Strain','Breakdown Type']
        error = 'Неверный формат разбивки данных'

    return template(allowed_file_types, data, columns, error)

def is_int(str):
    try:
        int(str)
        return True
    except ValueError:
         return False

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False



